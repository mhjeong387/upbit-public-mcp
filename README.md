# Upbit Public MCP Server

ChatGPT/MCP 클라이언트가 **업비트 Public REST API**를 직접 조회하도록 만든 read-only MCP 서버입니다.

## 포함 도구

- `health_check` — KRW-BTC로 업비트 연결 및 응답 timestamp 확인
- `get_markets` — 전체 마켓 목록
- `get_all_tickers` — `/v1/ticker/all` (KRW/BTC/USDT)
- `get_ticker` — 지정 종목 현재가
- `get_orderbook` — 호가 + top10 imbalance
- `get_candles` — 1초/분/일/주/월/연 캔들
- `get_recent_trades` — 최근 체결
- `analyze_market` — 5m/15m/60m/240m/1d + RSI14/EMA20·60/MACD/BB/ATR/거래량 + 호가
- `scan_krw_market` — `/ticker/all` 기반 KRW 전체 스크리닝 후 후보 기술분석

`scan_krw_market`의 점수는 기술적 **스크리닝 휴리스틱**이며 미래 수익률 확률이나 보장을 뜻하지 않습니다.

## 1) 로컬 실행

Python 3.10+ 권장.

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS/Linux
source .venv/bin/activate

pip install -r requirements.txt
python server.py
```

기본 MCP endpoint:

```text
http://localhost:8000/mcp
```

공식 MCP Inspector로 먼저 테스트하려면:

```bash
mcp dev server.py
```

## 2) 원격 배포

ChatGPT에서 접근하려면 일반적으로 인터넷에서 접근 가능한 **HTTPS Streamable HTTP MCP URL**이 필요합니다. 서버를 Railway/Render/Fly.io/VPS 등 ASGI/Python 실행 환경에 배포하고 8000 포트를 서비스 포트에 연결하세요.

환경변수:

```text
PORT=8000
MCP_HOST=0.0.0.0
```

배포 후 endpoint 예시:

```text
https://YOUR-DOMAIN.example/mcp
```

## 3) ChatGPT 연결 후 예시 요청

```text
Upbit MCP의 scan_krw_market를 사용해서 KRW 전체를 스캔해.
top_n=5, shortlist_size=15, min_turnover_krw=3000000000.
각 후보는 현재가 timestamp와 5m/15m/60m/240m/1d 지표를 검증하고,
API 결과가 stale이면 후보에서 제외해.
```

개별 코인:

```text
Upbit MCP로 KRW-QUID, KRW-GEOD, KRW-0G를 analyze_market 해.
현재가/호가/5m/15m/1h/4h/일봉을 비교하고 데이터 시각도 표시해.
```

## 데이터 신선도 원칙

서버는 Upbit 응답의 `timestamp`를 이용해 `source_age_seconds`를 같이 반환합니다.
ChatGPT 분석 시 다음 원칙을 권장합니다.

1. ticker/orderbook은 `source_age_seconds` 확인
2. candle은 `latest_candle_kst` 및 `latest_source_age_seconds` 확인
3. 오래된 데이터는 현재가/실시간 분석에서 제외
4. 검색엔진 캐시나 데이터랩 값으로 API 실패값을 대체하지 않기

## 보안

이 프로젝트는 **Public API 전용**이라 Upbit Access Key/Secret Key가 필요 없습니다.
Private 주문/잔고 API를 이 서버에 추가할 경우 read-only와 trading 권한을 반드시 분리하고, Secret Key를 ChatGPT 채팅에 붙여넣지 마세요.

## 참고

Upbit Public REST base URL:

```text
https://api.upbit.com
```

MCP는 Streamable HTTP로 실행됩니다.
