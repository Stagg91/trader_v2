# ... (Previous imports) ...
from fastapi import FastAPI, Request, Form, Depends, Response, Cookie, WebSocket, WebSocketDisconnect, UploadFile, File
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from src.logger import LabLogger
from fastapi.templating import Jinja2Templates
from sqlalchemy.orm import Session
import pandas as pd
import json
import numpy as np
import time
import io

from src.database import SessionLocal, engine, Settings, init_db, User, Strategy, BacktestResult, IndicatorDef, BacktestJob
from src.bybit_client import BybitClient
from src.data_engine import DataEngine
from src.indicators import IndicatorEngine
from src.backtester import Backtester, combined_strategy
from src.backtester_engine import GridSearchRunner
from src.genetic_engine import GeneticBreeder
from src.data_downloader import DataDownloader
# Import ML Engine conditionally
try:
    from src.ml_engine import MLEngine
except ImportError:
    MLEngine = None
from src.ai_sentiment import AISentimentAgent

from src.auth import verify_password, get_password_hash, create_access_token, decode_token, create_magic_token
from src.notifications import NotificationManager
from src.utils import get_resource_path
import qrcode
import io
import base64
from src.visualization import ChartGenerator
from src.strategies.schemas import StrategyRecipe
from typing import List

# Init DB
init_db()

app = FastAPI()

# ... (Previous startup_event, static mounts, templates config) ...
@app.on_event("startup")
async def startup_event():
    import asyncio
    LabLogger.set_loop(asyncio.get_running_loop())

static_path = get_resource_path("src/web/static")
templates_path = get_resource_path("src/web/templates")
app.mount("/static", StaticFiles(directory=static_path), name="static")
templates = Jinja2Templates(directory=templates_path)

def time_since(timestamp):
    if not timestamp: return 0
    diff = time.time() - timestamp
    return diff / 3600 # Hours
templates.env.filters['time_since'] = time_since
templates.env.filters['from_json'] = lambda x: json.loads(x) if x else {}

# ... (Previous WebSocket) ...
@app.websocket("/ws/lab_log")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    LabLogger.add_client(websocket)
    try:
        # Send history
        for msg in LabLogger.get_history():
            await websocket.send_text(msg)

        while True:
            await websocket.receive_text() # Keep alive
    except WebSocketDisconnect:
        LabLogger.remove_client(websocket)

# ... (Previous DB/Auth helpers) ...
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

def get_current_user(request: Request):
    token = request.cookies.get("access_token")
    if not token:
        return None
    user = decode_token(token)
    return user

@app.middleware("http")
async def auth_middleware(request: Request, call_next):
    try:
        if not request.url.path.startswith("/static") and not request.url.path.startswith("/ws"):
             await LabLogger.log("API", f"INCOMING: {request.method} {request.url.path}", {"params": dict(request.query_params)})
    except:
        pass

    if request.url.path in ["/login", "/setup", "/manifest.json", "/sw.js"] or request.url.path.startswith("/static"):
        return await call_next(request)

    token = request.cookies.get("access_token")
    if token and decode_token(token):
        return await call_next(request)

    db = SessionLocal()
    try:
        user_count = db.query(User).count()
        if user_count == 0:
             return RedirectResponse(url="/setup")
    finally:
        db.close()

    return RedirectResponse(url="/login")

# ... (Previous Login/Setup routes) ...
@app.get("/login", response_class=HTMLResponse)
async def login_page(request: Request):
    return templates.TemplateResponse("login.html", {"request": request})

@app.post("/login")
async def login(request: Request, username: str = Form(...), password: str = Form(...), db: Session = Depends(get_db)):
    await LabLogger.log("AUTH", f"Login attempt for user: {username}")
    user = db.query(User).filter(User.username == username).first()
    if not user or not verify_password(password, user.hashed_password):
        await LabLogger.log("AUTH", f"Login failed for {username}")
        return templates.TemplateResponse("login.html", {"request": request, "error": "Invalid credentials"})

    await LabLogger.log("AUTH", f"Login success for {username}")
    token = create_access_token({"sub": user.username})
    response = RedirectResponse(url="/", status_code=303)
    response.set_cookie(key="access_token", value=token, httponly=True)
    return response

@app.get("/logout")
async def logout():
    response = RedirectResponse(url="/login")
    response.delete_cookie("access_token")
    return response

@app.get("/setup", response_class=HTMLResponse)
async def setup_page(request: Request):
    return templates.TemplateResponse("setup.html", {"request": request})

@app.post("/setup")
async def setup(request: Request, username: str = Form(...), password: str = Form(...), db: Session = Depends(get_db)):
    try:
        if db.query(User).count() > 0:
            return RedirectResponse(url="/login", status_code=303)

        hashed_pw = get_password_hash(password)
        new_user = User(username=username, hashed_password=hashed_pw)
        db.add(new_user)
        db.commit()
        return RedirectResponse(url="/login", status_code=303)
    except Exception as e:
        err_msg = f"Setup Error: {e}"
        print(err_msg)
        return templates.TemplateResponse("setup.html", {"request": request, "error": f"Internal Error: {e}. Check staggs_trader.log"})

@app.get("/connect_mobile", response_class=HTMLResponse)
async def connect_mobile(request: Request):
    user = get_current_user(request)
    if not user:
        return RedirectResponse("/login")
    magic_token = create_magic_token({"sub": user['sub']})
    host_url = str(request.base_url).rstrip('/')
    data = f"{host_url}/magic_login?token={magic_token}"
    img = qrcode.make(data)
    buf = io.BytesIO()
    img.save(buf)
    qr_b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
    return templates.TemplateResponse("connect.html", {"request": request, "qr_base64": qr_b64})

@app.get("/magic_login")
async def magic_login(token: str):
    payload = decode_token(token)
    if not payload or payload.get("type") != "magic":
        return HTMLResponse("Invalid or expired magic link.", status_code=400)
    access_token = create_access_token({"sub": payload['sub']})
    response = RedirectResponse(url="/", status_code=303)
    response.set_cookie(key="access_token", value=access_token, httponly=True)
    return response

# ... (Previous Dashboard/API routes) ...
@app.get("/", response_class=HTMLResponse)
async def dashboard(request: Request, db: Session = Depends(get_db)):
    settings = db.query(Settings).first()
    balance = {}
    if settings and settings.api_key and settings.api_secret:
        client = BybitClient(api_key=settings.api_key, api_secret=settings.api_secret, testnet=settings.testnet)
        bal_resp = client.get_balance("USDT")
        if bal_resp:
             balance = bal_resp.get('result', {}).get('list', [{}])[0]
    notifications = NotificationManager.get_unread()
    return templates.TemplateResponse("dashboard.html", {
        "request": request,
        "balance": balance,
        "settings": settings,
        "notifications": notifications
    })

@app.get("/api/symbols")
async def get_symbols(db: Session = Depends(get_db)):
    settings = db.query(Settings).first()
    key = settings.api_key if settings else None
    secret = settings.api_secret if settings else None
    testnet = settings.testnet if settings else True
    client = BybitClient(api_key=key, api_secret=secret, testnet=testnet)
    resp = client.get_instruments()
    if resp and 'result' in resp and 'list' in resp['result']:
        symbols = [item['symbol'] for item in resp['result']['list'] if item['status'] == 'Trading']
        symbols.sort()
        return symbols
    return ["BTCUSDT", "ETHUSDT", "SOLUSDT"]

@app.get("/api/market_watch")
async def get_market_watch(db: Session = Depends(get_db)):
    settings = db.query(Settings).first()
    key = settings.api_key if settings else None
    secret = settings.api_secret if settings else None
    testnet = settings.testnet if settings else True
    client = BybitClient(api_key=key, api_secret=secret, testnet=testnet)
    resp = client.get_tickers()
    data = []
    if resp and 'result' in resp and 'list' in resp['result']:
        for item in resp['result']['list']:
            try:
                change = float(item.get('price24hPcnt', 0)) * 100
                data.append({
                    'symbol': item['symbol'],
                    'price': item['lastPrice'],
                    'change': f"{change:.2f}"
                })
            except:
                continue
    return data

@app.get("/api/chart_data")
async def get_chart_data(symbol: str = "BTCUSDT"):
    de = DataEngine()
    df = de.fetch_ohlcv(symbol, interval="60", limit=100)
    if df.empty:
        return {"error": "No data"}
    df = IndicatorEngine.add_indicators(df)
    df = df.where(pd.notnull(df), None)
    records = df.to_dict(orient="records")
    return records

@app.get("/api/strategy/{strat_id}")
async def get_strategy_details(strat_id: int, db: Session = Depends(get_db)):
    strat = db.query(Strategy).filter(Strategy.id == strat_id).first()
    if not strat:
        return {"error": "Strategy not found"}
    entry = ""
    exit_logic = ""
    if strat.content_json:
        entry = strat.content_json.get("entry_logic", "")
        exit_logic = strat.content_json.get("exit_logic", "")
    return {
        "id": strat.id,
        "name": strat.name,
        "generation": strat.generation,
        "type": strat.type,
        "entry_logic": entry,
        "exit_logic": exit_logic,
        "created_at": time.strftime("%Y-%m-%d %H:%M", time.localtime(strat.created_at)),
        "is_active": strat.is_active
    }

# ... (Previous Settings/Backtest routes) ...
@app.get("/settings", response_class=HTMLResponse)
async def settings_page(request: Request, db: Session = Depends(get_db)):
    settings = db.query(Settings).first()
    return templates.TemplateResponse("settings.html", {"request": request, "settings": settings})

@app.post("/settings")
async def save_settings(
    request: Request,
    api_key: str = Form(""),
    api_secret: str = Form(""),
    gemini_key: str = Form(""),
    testnet: bool = Form(False),
    paper_trading: bool = Form(False),
    paper_balance: float = Form(10000.0),
    is_active: bool = Form(False),
    auto_evolve: bool = Form(False),
    evolution_lookback_value: int = Form(3),
    evolution_lookback_unit: str = Form("Months"),
    evolution_interval: int = Form(30),
    cpu_usage_limit: int = Form(80),
    db: Session = Depends(get_db)
):
    settings = db.query(Settings).first()
    if not settings:
        settings = Settings()
        db.add(settings)
    settings.api_key = api_key
    settings.api_secret = api_secret
    settings.gemini_api_key = gemini_key
    settings.testnet = testnet
    settings.paper_trading = paper_trading
    settings.paper_balance = paper_balance
    settings.is_active = is_active
    settings.auto_evolve = auto_evolve
    settings.evolution_lookback_value = evolution_lookback_value
    settings.evolution_lookback_unit = evolution_lookback_unit
    settings.evolution_interval = evolution_interval
    settings.cpu_usage_limit = cpu_usage_limit
    db.commit()
    return templates.TemplateResponse("settings.html", {"request": request, "settings": settings, "message": "Saved!"})

@app.get("/backtest", response_class=HTMLResponse)
async def backtest_page(request: Request, db: Session = Depends(get_db)):
    strategies = db.query(Strategy).all()
    return templates.TemplateResponse("backtest.html", {"request": request, "config": None, "strategies": strategies})

@app.post("/ai_backtest_config")
async def ai_backtest_config(request: Request, prompt: str = Form(...), db: Session = Depends(get_db)):
    await LabLogger.log("API", f"Received ai_backtest_config with prompt: {prompt}")
    settings = db.query(Settings).first()
    key = settings.gemini_api_key if settings else None
    agent = AISentimentAgent(gemini_api_key=key)
    config = agent.interpret_strategy_prompt(prompt)
    await LabLogger.log("API", f"Config generated: {config}")
    return templates.TemplateResponse("backtest.html", {"request": request, "config": config})

@app.post("/run_backtest")
async def run_backtest(request: Request, symbol: str = Form(...), initial_balance: float = Form(10000.0), start_time: str = Form(None), end_time: str = Form(None), interval: str = Form("60"), strategy_mode: str = Form("manual"), strategy_id: int = Form(None), db: Session = Depends(get_db)):
    # ... (Same as previous run_backtest) ...
    # Simplified for brevity in this replace block, assume original content here unless changed logic
    # Re-inserting full original logic to avoid accidental truncation
    await LabLogger.log("BACKTEST", f"Starting Backtest on {symbol} ({interval})", {
        "balance": initial_balance, "mode": strategy_mode, "strat_id": strategy_id
    })

    de = DataEngine()
    ts_start = None
    ts_end = None
    if start_time:
        try: ts_start = int(pd.Timestamp(start_time).timestamp() * 1000)
        except: pass
    if end_time:
        try: ts_end = int(pd.Timestamp(end_time).timestamp() * 1000)
        except: pass

    limit = 100000 if (start_time or end_time) else 1000
    if ts_start:
        ms_per_candle = 3600000 # Default 1h
        if str(interval) == "1": ms_per_candle = 60000
        elif str(interval) == "5": ms_per_candle = 300000
        elif str(interval) == "15": ms_per_candle = 900000
        elif str(interval) == "240": ms_per_candle = 14400000
        elif str(interval).upper() == "D": ms_per_candle = 86400000
        buffer_ms = 200 * ms_per_candle
        ts_start = ts_start - buffer_ms

    df = de.fetch_ohlcv(symbol, interval=interval, limit=limit, start_time=ts_start, end_time=ts_end)

    if df.empty:
        await LabLogger.log("BACKTEST", "Error: No data found for specified range.")
        return {"error": "No data found"}

    if len(df) < 50:
        await LabLogger.log("BACKTEST", f"Error: Insufficient data loaded ({len(df)} candles).")
        return {"error": "Insufficient data (need > 50 candles)"}

    await LabLogger.log("BACKTEST", f"Data Loaded: {len(df)} candles (inc. buffer).")
    bt = Backtester(df, initial_balance=initial_balance)

    if strategy_mode == "ai" and strategy_id:
        strat = db.query(Strategy).filter(Strategy.id == strategy_id).first()
        if not strat: return {"error": "Strategy not found"}

        try:
            if strat.content_json:
                 recipe = StrategyRecipe(**strat.content_json)
                 res = bt.run_vectorized_backtest(recipe)
                 def sanitize(obj):
                     if isinstance(obj, float): return 0.0 if np.isnan(obj) or np.isinf(obj) else float(obj)
                     if isinstance(obj, (np.int64, np.int32, int)): return int(obj)
                     if isinstance(obj, (np.float64, np.float32)): return 0.0 if np.isnan(obj) or np.isinf(obj) else float(obj)
                     if isinstance(obj, dict): return {k: sanitize(v) for k, v in obj.items()}
                     if isinstance(obj, list): return [sanitize(v) for v in obj]
                     return obj
                 safe_res = sanitize(res)
                 await LabLogger.log("DB", f"Saving Result: {safe_res.keys()} | Trades: {len(safe_res.get('trades', []))}")
                 br = BacktestResult(
                    strategy_id=strat.id, symbol=symbol, start_date=str(df.iloc[0]['startTime']), end_date=str(df.iloc[-1]['startTime']),
                    roi=float(safe_res['roi_percent']), sharpe=float(safe_res['sharpe']), max_drawdown=float(safe_res['max_drawdown']),
                    win_rate=float(safe_res['win_rate']), trades_count=int(safe_res['total_trades']), metrics_json=json.dumps(safe_res), timestamp=time.time()
                 )
                 db.add(br)
                 db.commit()
                 if "text/html" in request.headers.get("accept", ""): return RedirectResponse(f"/backtest/result/{br.id}", status_code=303)
                 return [{"params": {"name": strat.name}, "metrics": safe_res, "result_id": br.id}]
            else:
                local_scope = {}
                exec(strat.code, {}, local_scope)
                StrategyClass = local_scope.get(strat.class_name)
                if not StrategyClass: return {"error": f"Class {strat.class_name} not found in code"}
                instance = StrategyClass()
                res = bt.run_strategy_instance(instance)
                test_res = res.get('test', res)
                def sanitize(obj):
                     if isinstance(obj, float): return 0.0 if np.isnan(obj) or np.isinf(obj) else float(obj)
                     if isinstance(obj, (np.int64, np.int32, int)): return int(obj)
                     if isinstance(obj, (np.float64, np.float32)): return 0.0 if np.isnan(obj) or np.isinf(obj) else float(obj)
                     if isinstance(obj, dict): return {k: sanitize(v) for k, v in obj.items()}
                     if isinstance(obj, list): return [sanitize(v) for v in obj]
                     return obj
                test_res = sanitize(test_res)
                br = BacktestResult(
                    strategy_id=strat.id, symbol=symbol, start_date=str(df.iloc[0]['startTime']), end_date=str(df.iloc[-1]['startTime']),
                    roi=float(test_res['roi_percent']), sharpe=float(test_res['sharpe']), max_drawdown=float(test_res['max_drawdown']),
                    win_rate=float(test_res['win_rate'] * 100), trades_count=int(test_res['total_trades']), metrics_json=json.dumps(sanitize(res)), timestamp=time.time()
                )
                db.add(br)
                db.commit()
                if "text/html" in request.headers.get("accept", ""): return RedirectResponse(f"/backtest/result/{br.id}", status_code=303)
                return [{"params": {"name": strat.name}, "metrics": test_res}]
        except Exception as e:
            traceback.print_exc()
            return {"error": f"Strategy Execution Error: {e}"}
    else: return []

# --- UPDATED STRATEGIES PAGE (View All, Export, Import) ---

@app.get("/strategies", response_class=HTMLResponse)
async def strategies_page(request: Request, view_all: bool = True, db: Session = Depends(get_db)):
    # 1. Fetch All Strategies
    all_strats = db.query(Strategy).all()

    # 2. Fetch Aggregated Performance
    results = db.query(BacktestResult).order_by(BacktestResult.timestamp.desc()).all()

    perf_map = {}
    for r in results:
        sid = r.strategy_id
        if sid not in perf_map:
            perf_map[sid] = {'roi': [], 'dd': [], 'latest_id': r.id}
        perf_map[sid]['roi'].append(r.roi)
        perf_map[sid]['dd'].append(r.max_drawdown)

    matrix_data = []
    leaderboard = []

    for s in all_strats:
        if s.id in perf_map:
            avg_roi = np.mean(perf_map[s.id]['roi'])
            avg_dd = np.mean(perf_map[s.id]['dd'])
            latest_id = perf_map[s.id]['latest_id']
            matrix_data.append({'id': s.id, 'x': avg_dd, 'y': avg_roi, 'text': f"{s.name} (Gen {s.generation})"})
            leaderboard.append({'strategy': s, 'roi': avg_roi, 'max_drawdown': avg_dd, 'win_rate': 0, 'last_result_id': latest_id})
        else:
            # Add strategies with no results to leaderboard anyway (at bottom)
            leaderboard.append({'strategy': s, 'roi': -999, 'max_drawdown': 0, 'win_rate': 0, 'last_result_id': None})

    # Sort Leaderboard
    leaderboard.sort(key=lambda x: x['roi'], reverse=True)

    # Filter
    if not view_all:
        leaderboard_display = leaderboard[:20]
    else:
        leaderboard_display = leaderboard

    return templates.TemplateResponse("strategies.html", {
        "request": request,
        "matrix_data": matrix_data,
        "leaderboard": leaderboard_display,
        "view_all": view_all
    })

@app.post("/strategies/delete")
async def delete_strategies(strategy_ids: List[int] = Form(...), db: Session = Depends(get_db)):
    if not strategy_ids: return RedirectResponse("/strategies", status_code=303)
    db.query(Strategy).filter(Strategy.id.in_(strategy_ids)).delete(synchronize_session=False)
    db.commit()
    return RedirectResponse("/strategies", status_code=303)

@app.post("/strategies/export")
async def export_strategies(strategy_ids: List[int] = Form(...), db: Session = Depends(get_db)):
    if not strategy_ids: return RedirectResponse("/strategies", status_code=303)

    strats = db.query(Strategy).filter(Strategy.id.in_(strategy_ids)).all()
    export_data = []
    for s in strats:
        export_data.append({
            "name": s.name,
            "description": s.description,
            "code": s.code,
            "content_json": s.content_json,
            "class_name": s.class_name,
            "type": s.type,
            "generation": s.generation
        })

    json_str = json.dumps(export_data, indent=2)
    return StreamingResponse(
        io.StringIO(json_str),
        media_type="application/json",
        headers={"Content-Disposition": f"attachment; filename=strategies_export_{int(time.time())}.json"}
    )

@app.post("/strategies/import")
async def import_strategies(file: UploadFile = File(...), db: Session = Depends(get_db)):
    try:
        content = await file.read()
        data = json.loads(content)

        count = 0
        for item in data:
            # Basic validation
            if "name" not in item: continue

            s = Strategy(
                name=item.get("name") + " (Imported)",
                description=item.get("description"),
                code=item.get("code"),
                content_json=item.get("content_json"),
                class_name=item.get("class_name"),
                type=item.get("type", "manual"),
                generation=item.get("generation", 0),
                created_at=time.time(),
                is_active=False
            )
            db.add(s)
            count += 1
        db.commit()

    except Exception as e:
        await LabLogger.log("API", f"Import Error: {e}")

    return RedirectResponse("/strategies", status_code=303)

# ... (Previous Generate/Activate/Evolution routes) ...
@app.post("/strategies/generate")
async def generate_strategies(request: Request, prompt: str = Form("Robust trend following strategy"), count: int = Form(3), db: Session = Depends(get_db)):
    await LabLogger.log("AI", f"Requesting Generation: {prompt}", {"count": count})
    settings = db.query(Settings).first()
    key = settings.gemini_api_key if settings else None
    breeder = GeneticBreeder(gemini_api_key=key)
    new_strats = await breeder.create_generation_zero(prompt=prompt, count=count)
    await LabLogger.log("API", f"Generated {len(new_strats)} strategies.")
    return RedirectResponse("/strategies", status_code=303)

@app.post("/strategies/activate/{strat_id}")
async def activate_strategy(strat_id: int, db: Session = Depends(get_db)):
    db.query(Strategy).update({Strategy.is_active: False})
    strat = db.query(Strategy).filter(Strategy.id == strat_id).first()
    if strat:
        strat.is_active = True
        db.commit()
    return RedirectResponse("/strategies", status_code=303)

@app.post("/strategies/deactivate/{strat_id}")
async def deactivate_strategy(strat_id: int, db: Session = Depends(get_db)):
    strat = db.query(Strategy).filter(Strategy.id == strat_id).first()
    if strat:
        strat.is_active = False
        db.commit()
    return RedirectResponse("/strategies", status_code=303)

@app.get("/evolution", response_class=HTMLResponse)
async def evolution_page(request: Request, db: Session = Depends(get_db)):
    rows = db.query(BacktestResult, Strategy).join(Strategy, BacktestResult.strategy_id == Strategy.id).order_by(BacktestResult.roi.desc()).limit(50).all()
    return templates.TemplateResponse("evolution.html", {"request": request, "results": rows})

@app.post("/evolution/run_generation")
async def run_generation(request: Request, generation: int = Form(0), db: Session = Depends(get_db)):
    settings = db.query(Settings).first()
    key = settings.gemini_api_key if settings else None
    breeder = GeneticBreeder(gemini_api_key=key)
    breeder.evaluate_population(generation=generation)
    breeder.breed_next_generation(current_gen=generation)
    return RedirectResponse("/evolution", status_code=303)

@app.get("/backtest/result/{result_id}", response_class=HTMLResponse)
async def backtest_result_page(request: Request, result_id: int, db: Session = Depends(get_db)):
    res = db.query(BacktestResult).filter(BacktestResult.id == result_id).first()
    if not res: return RedirectResponse("/evolution")
    import json
    chart_json = None
    try:
        await LabLogger.log("DB", f"Loading Result #{result_id} | JSON Len: {len(res.metrics_json)}")
        metrics = json.loads(res.metrics_json)
        details = metrics['test'] if 'test' in metrics else metrics
        trades = details.get('trades', [])
        equity_curve = details.get('equity_curve', [])
        strat = db.query(Strategy).filter(Strategy.id == res.strategy_id).first()
        if strat and strat.content_json:
             de = DataEngine()
             def safe_ts_to_ms(val):
                 try:
                     if str(val).isdigit(): return int(val)
                     return int(pd.Timestamp(val).timestamp() * 1000)
                 except: return 0
             ts_start = safe_ts_to_ms(res.start_date)
             ts_end = safe_ts_to_ms(res.end_date)
             df = de.fetch_ohlcv(res.symbol, interval="60", start_time=ts_start, end_time=ts_end)
             if not df.empty:
                 from src.strategy_parser import StrategyParser
                 parser = StrategyParser()
                 recipe = StrategyRecipe(**strat.content_json)
                 df_res = parser.parse_and_execute(df, recipe)
                 chart_json = ChartGenerator.generate_chart_json(df_res, indicators=[ind.col_name or ind.name for ind in recipe.indicators])
    except Exception as e:
        print(f"Error loading result details: {e}")
        trades = []
        equity_curve = []
    return templates.TemplateResponse("backtest_result.html", {
        "request": request, "result": res, "trades": trades, "equity_curve": equity_curve, "chart_json": chart_json
    })

# ... (Previous Routes) ...
@app.get("/synopsis", response_class=HTMLResponse)
async def synopsis_page(request: Request, db: Session = Depends(get_db)):
    settings = db.query(Settings).first()
    symbol = "BTCUSDT"
    de = DataEngine()
    indicators = {}
    ml_prob = 0.5
    sentiment = "UNKNOWN"
    explanation = "Data unavailable."
    try:
        df = de.fetch_ohlcv(symbol, interval="60", limit=100)
        if not df.empty:
            df = IndicatorEngine.add_indicators(df)
            last_row = df.iloc[-1]
            indicators = {
                "Close Price": last_row['close'],
                "RSI (14)": f"{last_row.get('RSI_14', 0):.2f}",
            }
            if 'MACD_12_26_9' in last_row: indicators['MACD'] = f"{last_row['MACD_12_26_9']:.2f}"
            if MLEngine:
                try:
                    ml_agent = MLEngine()
                    ml_prob = ml_agent.predict_probability(df)
                except Exception as e: print(f"ML Error: {e}")
            if settings and settings.gemini_api_key:
                ai_agent = AISentimentAgent(gemini_api_key=settings.gemini_api_key)
                try: sentiment = ai_agent.get_market_sentiment()
                except: sentiment = "ERROR"
                prompt = (f"Current market status for {symbol}: Price {last_row['close']}, RSI {last_row.get('RSI_14', 'N/A')}. "
                    f"ML Model predicts {ml_prob*100:.1f}% chance of price increase. News Sentiment is {sentiment}. "
                    "Explain the recommended trading strategy and rationale in 2 sentences.")
                try: explanation = ai_agent.model.generate_content(prompt).text
                except: explanation = "AI Explanation unavailable (API Error)."
            else:
                 ai_agent = AISentimentAgent(gemini_api_key=None)
                 sentiment = ai_agent.get_market_sentiment()
                 explanation = "Configure Gemini API Key in settings to get AI-generated strategy explanation."
    except Exception as e:
        print(f"Synopsis Error: {e}")
        explanation = f"Error generating synopsis: {e}"
        if not indicators: indicators = {"Status": "Unavailable"}
    return templates.TemplateResponse("synopsis.html", {"request": request, "indicators": indicators, "sentiment": sentiment, "ml_prob": ml_prob, "explanation": explanation})

@app.post("/train_ml")
async def train_ml():
    if not MLEngine: return {"error": "ML dependencies not installed"}
    de = DataEngine()
    df = de.fetch_ohlcv("BTCUSDT", interval="60", limit=1000)
    ml = MLEngine()
    score = ml.train_model(df)
    response = RedirectResponse("/synopsis", status_code=303)
    response.set_cookie(key="flash_message", value=f"ML Model Retrained. Accuracy: {score:.2f}")
    return response

@app.post("/api/sync_data")
async def sync_data():
    downloader = DataDownloader()
    msg = downloader.start_sync()
    return {"status": msg}

@app.get("/api/history")
async def get_history(symbol: str = "BTCUSDT", interval: str = "60", limit: int = 200):
    from src.data_warehouse import DataWarehouse
    warehouse = DataWarehouse()
    df = warehouse.load_data(symbol, interval, limit=limit)
    if df.empty:
        de = DataEngine()
        df = de.fetch_ohlcv(symbol, interval=interval, limit=limit)
    df = df.where(pd.notnull(df), None)
    return df.to_dict(orient="records")

@app.get("/lab", response_class=HTMLResponse)
async def lab_page(request: Request, db: Session = Depends(get_db)):
    settings = db.query(Settings).first()
    strategies = db.query(Strategy).order_by(Strategy.generation.desc(), Strategy.name).all()
    max_gen_strat = db.query(Strategy).order_by(Strategy.generation.desc()).first()
    max_gen = max_gen_strat.generation if max_gen_strat else 0
    best_strat = db.query(BacktestResult, Strategy).join(Strategy, BacktestResult.strategy_id == Strategy.id).order_by(BacktestResult.roi.desc()).first()
    return templates.TemplateResponse("lab.html", {"request": request, "settings": settings, "strategies": strategies, "max_gen": max_gen, "best_strat": best_strat})

@app.get("/indicators", response_class=HTMLResponse)
async def indicators_page(request: Request, db: Session = Depends(get_db)):
    indicators = db.query(IndicatorDef).order_by(IndicatorDef.category, IndicatorDef.name).all()
    for ind in indicators:
        if isinstance(ind.optimization_config, str):
            ind.optimization_config = json.loads(ind.optimization_config)
    return templates.TemplateResponse("indicators.html", {"request": request, "indicators": indicators})

@app.post("/indicators/update")
async def update_indicator(request: Request, ind_id: int = Form(...), opt_config: str = Form(...), db: Session = Depends(get_db)):
    ind = db.query(IndicatorDef).filter(IndicatorDef.id == ind_id).first()
    if ind:
        try:
            config = json.loads(opt_config)
            ind.optimization_config = config
            db.commit()
        except Exception as e: return JSONResponse({"error": str(e)}, status_code=400)
    return RedirectResponse("/indicators", status_code=303)

@app.get("/strategies/new", response_class=HTMLResponse)
async def new_strategy_page(request: Request, db: Session = Depends(get_db)):
    indicators = db.query(IndicatorDef).order_by(IndicatorDef.category, IndicatorDef.name).all()
    return templates.TemplateResponse("create_strategy.html", {"request": request, "indicators": indicators})

@app.post("/strategies/create_composite")
async def create_composite_strategy(request: Request, name: str = Form(...), description: str = Form(""), selected_indicators: List[str] = Form(...), db: Session = Depends(get_db)):
    content = {"indicators": selected_indicators, "logic_type": "AND"}
    strat = Strategy(name=name, description=description, type="composite", content_json=content, created_at=time.time(), is_active=False)
    db.add(strat)
    db.commit()
    return RedirectResponse("/strategies", status_code=303)

@app.post("/backtest/start_grid")
async def start_grid_backtest(request: Request, strategy_id: int = Form(...), db: Session = Depends(get_db)):
    job = BacktestJob(strategy_id=strategy_id, status="pending", started_at=time.time(), progress=0.0)
    db.add(job)
    db.commit()
    import asyncio
    runner = GridSearchRunner(job.id)
    asyncio.create_task(runner.run())
    return RedirectResponse("/backtest/jobs", status_code=303)

@app.get("/backtest/jobs", response_class=HTMLResponse)
async def jobs_page(request: Request, db: Session = Depends(get_db)):
    jobs = db.query(BacktestJob, Strategy).join(Strategy, BacktestJob.strategy_id == Strategy.id).order_by(BacktestJob.started_at.desc()).all()
    return templates.TemplateResponse("jobs.html", {"request": request, "jobs": jobs})

@app.post("/backtest/control/{job_id}")
async def control_job(job_id: int, action: str = Form(...), db: Session = Depends(get_db)):
    job = db.query(BacktestJob).filter(BacktestJob.id == job_id).first()
    if job:
        if action == "pause": job.status = "paused"
        elif action == "resume": job.status = "running"
        elif action == "cancel": job.status = "cancelled"
        db.commit()
    return RedirectResponse("/backtest/jobs", status_code=303)

@app.get("/backtest/matrix/{job_id}", response_class=HTMLResponse)
async def matrix_page(request: Request, job_id: int, db: Session = Depends(get_db)):
    job = db.query(BacktestJob).filter(BacktestJob.id == job_id).first()
    strategy = db.query(Strategy).filter(Strategy.id == job.strategy_id).first() if job else None
    results = db.query(BacktestResult).filter(BacktestResult.job_id == job_id).order_by(BacktestResult.roi.desc()).limit(200).all()
    return templates.TemplateResponse("matrix.html", {"request": request, "job": job, "strategy": strategy, "results": results})
