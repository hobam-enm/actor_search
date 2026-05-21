import io
import json
import time
from datetime import date, timedelta

import pandas as pd
import requests
import streamlit as st

DATALAB_URL = "https://openapi.naver.com/v1/datalab/search"

ANCHOR_MONTH_START = "2026-01-01"
ANCHOR_MONTH_END = "2026-01-31"

ANCHORS = [
    {"group": "anchor_tvn", "keyword": "tvN", "monthly_volume": 47600},
    {"group": "anchor_nf", "keyword": "넷플릭스", "monthly_volume": 2353200},
]

TARGET_GROUPS = {
    "기본검색량": ["{actor}"],
    "후행탐색검색량": [
        "{actor} 나무위키",
        "{actor} 인스타",
    ],
    "연애상대화검색량": [
        "{actor} 로맨스",
        "{actor} 멜로",
        "{actor} 케미",
        "{actor} 설렘",
        "{actor} 설렌다",
        "{actor} 심쿵",
        "{actor} 플러팅",
        "{actor} 눈빛",
        "{actor} 키스신",
        "{actor} 고백신",
        "{actor} 남친짤",
        "{actor} 남친미",
        "{actor} 남자친구",
        "{actor} 남친",
        "{actor} boyfriend",
        "{actor} 이상형",
        "{actor} 첫사랑",
        "{actor} 짝사랑",
        "{actor} 여자친구",
        "{actor} 여친",
    ],
}


def get_secret(key: str) -> str:
    try:
        return str(st.secrets[key]).strip()
    except Exception:
        return ""


def post_datalab(start_date: str, end_date: str, keyword_groups: list[dict], time_unit: str = "date") -> dict:
    client_id = get_secret("NAVER_CLIENT_ID")
    client_secret = get_secret("NAVER_CLIENT_SECRET")

    if not client_id or not client_secret:
        raise RuntimeError("Streamlit Secrets에 NAVER_CLIENT_ID / NAVER_CLIENT_SECRET가 없습니다.")

    headers = {
        "X-Naver-Client-Id": client_id,
        "X-Naver-Client-Secret": client_secret,
        "Content-Type": "application/json",
    }

    payload = {
        "startDate": start_date,
        "endDate": end_date,
        "timeUnit": time_unit,
        "keywordGroups": keyword_groups,
    }

    resp = requests.post(
        DATALAB_URL,
        headers=headers,
        data=json.dumps(payload, ensure_ascii=False),
        timeout=30,
    )

    if resp.status_code != 200:
        raise RuntimeError(f"[DATALAB] HTTP {resp.status_code}\n{resp.text}")

    return resp.json()


def datalab_json_to_pivot(api_json: dict) -> pd.DataFrame:
    rows = []
    for result in api_json.get("results", []):
        title = result.get("title")
        for point in result.get("data", []):
            rows.append({"date": point.get("period"), "group": title, "ratio": point.get("ratio", 0)})

    df = pd.DataFrame(rows)
    if df.empty:
        return pd.DataFrame()

    df["date"] = pd.to_datetime(df["date"])
    return df.pivot_table(index="date", columns="group", values="ratio", aggfunc="first").sort_index().fillna(0)


def compute_k_from_anchor_month(piv: pd.DataFrame, anchor_group: str, monthly_volume: float) -> float:
    jan_mask = (piv.index >= pd.to_datetime(ANCHOR_MONTH_START)) & (piv.index <= pd.to_datetime(ANCHOR_MONTH_END))
    if anchor_group not in piv.columns:
        raise RuntimeError(f"DataLab 결과에 앵커 그룹 '{anchor_group}' 컬럼이 없습니다.")
    ratio_sum = float(piv.loc[jan_mask, anchor_group].sum())
    if ratio_sum <= 0:
        raise RuntimeError(f"앵커 '{anchor_group}'의 기준월 ratio 합이 0입니다.")
    return monthly_volume / ratio_sum


def build_keyword_groups(actor: str) -> list[dict]:
    groups = []
    for anchor in ANCHORS:
        groups.append({"groupName": anchor["group"], "keywords": [anchor["keyword"]]})
    for group_name, patterns in TARGET_GROUPS.items():
        groups.append({"groupName": group_name, "keywords": [p.format(actor=actor) for p in patterns]})
    return groups


def estimate_actor_abs_daily(actor: str, user_start: str, user_end: str) -> pd.DataFrame:
    user_start_dt = pd.to_datetime(user_start)
    user_end_dt = pd.to_datetime(user_end)
    anchor_start_dt = pd.to_datetime(ANCHOR_MONTH_START)
    anchor_end_dt = pd.to_datetime(ANCHOR_MONTH_END)

    api_start = min(user_start_dt, anchor_start_dt).strftime("%Y-%m-%d")
    api_end = max(user_end_dt, anchor_end_dt).strftime("%Y-%m-%d")

    api_json = post_datalab(api_start, api_end, build_keyword_groups(actor), time_unit="date")
    piv = datalab_json_to_pivot(api_json)
    if piv.empty:
        raise RuntimeError("DataLab 결과가 비었습니다.")

    k_map = {a["group"]: compute_k_from_anchor_month(piv, a["group"], a["monthly_volume"]) for a in ANCHORS}
    total_anchor_volume = sum(a["monthly_volume"] for a in ANCHORS)
    weights = {a["group"]: a["monthly_volume"] / total_anchor_volume for a in ANCHORS}

    out = pd.DataFrame(index=piv.index)
    out["actor"] = actor

    for target_group in TARGET_GROUPS.keys():
        if target_group not in piv.columns:
            out[target_group] = 0.0
            continue
        est = 0.0
        for anchor in ANCHORS:
            anchor_group = anchor["group"]
            est += weights[anchor_group] * (piv[target_group] * k_map[anchor_group])
        out[target_group] = est

    out = out.reset_index().rename(columns={"index": "date"})
    return out[(out["date"] >= user_start_dt) & (out["date"] <= user_end_dt)].copy()


def aggregate_result(df_daily: pd.DataFrame, view_unit: str) -> pd.DataFrame:
    df = df_daily.copy()
    if df.empty:
        return pd.DataFrame()

    if view_unit == "일자별":
        df["기간"] = df["date"].dt.strftime("%Y-%m-%d")
    elif view_unit == "주차별":
        df["week_start"] = df["date"] - pd.to_timedelta(df["date"].dt.dayofweek, unit="d")
        df["기간"] = df["week_start"].dt.strftime("%Y-%m-%d") + " 주차"
    else:
        df["기간"] = df["date"].dt.to_period("M").astype(str)

    result = df.groupby(["actor", "기간"])[list(TARGET_GROUPS.keys())].sum().reset_index()
    for col in TARGET_GROUPS.keys():
        result[col] = result[col].round().astype(int)

    return result.rename(columns={
        "actor": "배우명",
        "기본검색량": "기본 검색량",
        "후행탐색검색량": "후행탐색 검색량",
        "연애상대화검색량": "연애상대화 검색량",
    })


def make_period_total_summary(df_agg: pd.DataFrame) -> pd.DataFrame:
    if df_agg.empty:
        return pd.DataFrame()

    summary = df_agg.groupby("배우명", as_index=False)[["기본 검색량", "후행탐색 검색량", "연애상대화 검색량"]].sum()
    summary["후행탐색/기본"] = (summary["후행탐색 검색량"] / summary["기본 검색량"].replace(0, pd.NA)).fillna(0)
    summary["연애상대화/기본"] = (summary["연애상대화 검색량"] / summary["기본 검색량"].replace(0, pd.NA)).fillna(0)
    summary = summary.sort_values(["후행탐색 검색량", "연애상대화 검색량", "기본 검색량"], ascending=False)
    summary["후행탐색/기본"] = (summary["후행탐색/기본"] * 100).round(2).astype(str) + "%"
    summary["연애상대화/기본"] = (summary["연애상대화/기본"] * 100).round(2).astype(str) + "%"
    return summary


def to_csv_bytes(df: pd.DataFrame) -> bytes:
    return df.to_csv(index=False, encoding="utf-8-sig").encode("utf-8-sig")


def to_excel_bytes(sheets: dict[str, pd.DataFrame]) -> bytes:
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine="xlsxwriter") as writer:
        for sheet_name, df in sheets.items():
            safe_sheet = sheet_name[:31]
            df.to_excel(writer, sheet_name=safe_sheet, index=False)
            ws = writer.sheets[safe_sheet]
            wb = writer.book
            comma_fmt = wb.add_format({"num_format": "#,##0"})
            for idx, col in enumerate(df.columns):
                width = max(12, min(36, len(str(col)) + 4))
                if "검색량" in str(col):
                    ws.set_column(idx, idx, width, comma_fmt)
                else:
                    ws.set_column(idx, idx, width)
    return output.getvalue()


st.set_page_config(page_title="남배우 검색량 수집기", page_icon="📈", layout="wide")
st.title("📈 남배우 검색량 수집기")
st.caption("네이버 DataLab 상대지수를 앵커 키워드 기준으로 절대검색량 추정치로 변환")

with st.expander("수집 검색어 구조", expanded=True):
    st.markdown(
        """
| 최종 컬럼 | 검색어 묶음 |
|---|---|
| 기본 검색량 | 배우명 |
| 후행탐색 검색량 | 배우명 나무위키, 배우명 인스타 |
| 연애상대화 검색량 | 배우명 로맨스, 배우명 멜로, 배우명 케미, 배우명 설렘, 배우명 설렌다, 배우명 심쿵, 배우명 플러팅, 배우명 눈빛, 배우명 키스신, 배우명 고백신, 배우명 남친짤, 배우명 남친미, 배우명 남자친구, 배우명 남친, 배우명 boyfriend, 배우명 이상형, 배우명 첫사랑, 배우명 짝사랑, 배우명 사귀고 싶다, 배우명 결혼하고 싶다, 배우명 여자친구, 배우명 여친 |
        """
    )
    st.caption("주의: 여자친구/여친은 연애상대화 신호와 사생활 탐색 신호가 섞일 수 있습니다.")

client_id = get_secret("NAVER_CLIENT_ID")
client_secret = get_secret("NAVER_CLIENT_SECRET")

if client_id and client_secret:
    st.success("NAVER_CLIENT_ID / NAVER_CLIENT_SECRET 시크릿이 감지되었습니다.")
else:
    st.error("Streamlit Cloud Secrets에 NAVER_CLIENT_ID / NAVER_CLIENT_SECRET를 등록하세요.")
    st.code('NAVER_CLIENT_ID = "발급받은_CLIENT_ID"\nNAVER_CLIENT_SECRET = "발급받은_CLIENT_SECRET"', language="toml")
    st.stop()

st.sidebar.header("수집 설정")
today = date.today()
default_end = today - timedelta(days=1)
default_start = default_end - timedelta(days=27)

start_date = st.sidebar.date_input("시작일", value=default_start)
end_date = st.sidebar.date_input("종료일", value=default_end)
view_unit = st.sidebar.selectbox("결과 집계 단위", ["주차별", "일자별", "월별"], index=0)
max_actors = st.sidebar.number_input("이번 실행 최대 배우 수", min_value=1, max_value=1000, value=800, step=50)
sleep_sec = st.sidebar.number_input("배우별 호출 간 대기시간(초)", min_value=0.0, max_value=3.0, value=0.15, step=0.05)

uploaded_file = st.file_uploader("배우 리스트 CSV 업로드", type=["csv"], help="actor 컬럼이 있으면 actor 컬럼을 사용하고, 없으면 첫 번째 컬럼을 배우명으로 사용합니다.")

sample_df = pd.DataFrame({"actor": ["김선호", "변우석", "추영우"]})
st.download_button("샘플 CSV 다운로드", data=to_csv_bytes(sample_df), file_name="sample_actors.csv", mime="text/csv")

actors = []
if uploaded_file is not None:
    try:
        actor_df = pd.read_csv(uploaded_file)
    except UnicodeDecodeError:
        uploaded_file.seek(0)
        actor_df = pd.read_csv(uploaded_file, encoding="cp949")

    actor_col = "actor" if "actor" in actor_df.columns else actor_df.columns[0]
    actors = actor_df[actor_col].dropna().astype(str).str.strip().loc[lambda s: s.ne("")].drop_duplicates().head(int(max_actors)).tolist()

    st.subheader("업로드된 배우 리스트")
    st.write(f"총 {len(actors):,}명")
    st.dataframe(pd.DataFrame({"actor": actors}), use_container_width=True, height=240)
else:
    st.info("CSV를 업로드하면 수집 버튼이 활성화됩니다.")

run = st.button("검색량 수집 시작", type="primary", disabled=not bool(actors))

if run:
    if start_date > end_date:
        st.error("시작일은 종료일보다 늦을 수 없습니다.")
        st.stop()

    st.warning(f"이번 실행 예상 API 호출 수: {len(actors):,}회. 배우 1명당 앵커 2개 + 타깃 3개, 총 5개 그룹을 한 번에 호출합니다.")

    daily_parts = []
    errors = []
    progress = st.progress(0)
    status = st.empty()
    start_str = start_date.strftime("%Y-%m-%d")
    end_str = end_date.strftime("%Y-%m-%d")

    for idx, actor in enumerate(actors, start=1):
        status.write(f"[{idx:,}/{len(actors):,}] {actor} 수집 중...")
        try:
            daily = estimate_actor_abs_daily(actor, start_str, end_str)
            daily_parts.append(daily)
        except Exception as e:
            errors.append({"배우명": actor, "오류": str(e)})
        progress.progress(idx / len(actors))
        if sleep_sec > 0:
            time.sleep(float(sleep_sec))

    if daily_parts:
        df_daily = pd.concat(daily_parts, ignore_index=True)
        df_agg = aggregate_result(df_daily, view_unit=view_unit)
        df_period_total = make_period_total_summary(df_agg)
    else:
        df_daily = pd.DataFrame()
        df_agg = pd.DataFrame()
        df_period_total = pd.DataFrame()

    df_errors = pd.DataFrame(errors)
    st.session_state["df_daily"] = df_daily
    st.session_state["df_agg"] = df_agg
    st.session_state["df_period_total"] = df_period_total
    st.session_state["df_errors"] = df_errors
    st.success("수집 완료")

if "df_agg" in st.session_state:
    df_daily = st.session_state["df_daily"]
    df_agg = st.session_state["df_agg"]
    df_period_total = st.session_state["df_period_total"]
    df_errors = st.session_state["df_errors"]

    st.header("결과")
    if df_agg.empty:
        st.error("정상 수집된 결과가 없습니다.")
    else:
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("수집 배우 수", f"{df_agg['배우명'].nunique():,}")
        c2.metric("결과 행 수", f"{len(df_agg):,}")
        c3.metric("기간 수", f"{df_agg['기간'].nunique():,}")
        c4.metric("오류 수", f"{len(df_errors):,}")

        st.subheader("기간 합산 요약")
        st.caption("선택한 전체 기간의 검색량을 배우별로 합산한 표입니다.")
        st.dataframe(df_period_total, use_container_width=True, height=480)

        st.subheader("기간별 결과")
        st.dataframe(df_agg, use_container_width=True, height=520)

        st.download_button("기간 합산 요약 CSV 다운로드", data=to_csv_bytes(df_period_total), file_name="actor_search_volume_period_total.csv", mime="text/csv", type="primary")
        st.download_button("기간별 결과 CSV 다운로드", data=to_csv_bytes(df_agg), file_name="actor_search_volume_by_period.csv", mime="text/csv")

        excel_bytes = to_excel_bytes({"기간합산요약": df_period_total, "기간별결과": df_agg, "일자별원천": df_daily, "오류": df_errors})
        st.download_button("엑셀 다운로드", data=excel_bytes, file_name="actor_search_volume_simple.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

    if not df_errors.empty:
        st.subheader("오류 목록")
        st.dataframe(df_errors, use_container_width=True)
