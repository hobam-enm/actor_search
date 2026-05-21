import time
from datetime import date, timedelta

import pandas as pd
import requests
import streamlit as st


# =========================
# 기본 설정
# =========================

API_URL = "https://openapi.naver.com/v1/datalab/search"

DEFAULT_GROUPS = {
    "기본관심도": ["{actor}"],
    "로맨스연상": [
        "{actor} 로맨스",
        "{actor} 멜로",
        "{actor} 케미",
        "{actor} 설렘",
    ],
    "연애상대화": [
        "{actor} 남친짤",
        "{actor} 남친미",
        "{actor} 이상형",
        "{actor} 첫사랑",
    ],
    "후행탐색": [
        "{actor} 나무위키",
        "{actor} 인스타",
    ],
    "사생활노이즈": [
        "{actor} 여자친구",
        "{actor} 열애",
        "{actor} 결혼",
    ],
}


# =========================
# 유틸 함수
# =========================

def get_secret(key: str) -> str:
    """Streamlit Cloud secrets에서 API 키를 읽는다."""
    try:
        value = st.secrets[key]
    except Exception:
        value = ""
    return str(value).strip()


def normalize_actor_name(name: str) -> str:
    """배우명 공백/결측 정리."""
    if pd.isna(name):
        return ""
    return str(name).strip()


def build_keyword_groups(actor: str) -> list[dict]:
    """배우명 1명에 대한 네이버 데이터랩 keywordGroups 생성."""
    groups = []
    for group_name, patterns in DEFAULT_GROUPS.items():
        keywords = [pattern.format(actor=actor) for pattern in patterns]
        groups.append({"groupName": group_name, "keywords": keywords})
    return groups


def call_naver_datalab(
    actor: str,
    start_date: str,
    end_date: str,
    time_unit: str,
    device: str | None = None,
    gender: str | None = None,
    ages: list[str] | None = None,
    max_retries: int = 2,
    sleep_sec: float = 0.15,
) -> dict:
    """네이버 데이터랩 통합검색어 트렌드 API 호출."""
    client_id = get_secret("NAVER_CLIENT_ID")
    client_secret = get_secret("NAVER_CLIENT_SECRET")

    if not client_id or not client_secret:
        raise RuntimeError(
            "Streamlit secrets에 NAVER_CLIENT_ID / NAVER_CLIENT_SECRET가 없습니다."
        )

    headers = {
        "X-Naver-Client-Id": client_id,
        "X-Naver-Client-Secret": client_secret,
        "Content-Type": "application/json",
    }

    payload = {
        "startDate": start_date,
        "endDate": end_date,
        "timeUnit": time_unit,
        "keywordGroups": build_keyword_groups(actor),
    }

    if device and device != "전체":
        payload["device"] = device  # pc / mo

    if gender and gender != "전체":
        payload["gender"] = gender  # m / f

    if ages:
        payload["ages"] = ages

    last_error = None

    for attempt in range(max_retries + 1):
        try:
            response = requests.post(
                API_URL,
                headers=headers,
                json=payload,
                timeout=20,
            )

            if response.status_code == 200:
                time.sleep(sleep_sec)
                return response.json()

            last_error = f"HTTP {response.status_code}: {response.text[:500]}"

            if response.status_code in [429, 500, 502, 503, 504]:
                time.sleep(1.5 * (attempt + 1))
                continue

            break

        except Exception as e:
            last_error = str(e)
            time.sleep(1.5 * (attempt + 1))

    raise RuntimeError(last_error or "알 수 없는 API 오류")


def parse_api_result(actor: str, api_result: dict) -> pd.DataFrame:
    """API 응답 JSON을 long format DataFrame으로 변환."""
    rows = []

    for group in api_result.get("results", []):
        group_name = group.get("title", "")

        for item in group.get("data", []):
            rows.append(
                {
                    "actor": actor,
                    "period": item.get("period"),
                    "group": group_name,
                    "ratio": item.get("ratio"),
                }
            )

    if not rows:
        return pd.DataFrame(columns=["actor", "period", "group", "ratio"])

    return pd.DataFrame(rows)


def make_summary(df_long: pd.DataFrame) -> pd.DataFrame:
    """배우×기간별 wide summary 및 파생점수 생성."""
    if df_long.empty:
        return pd.DataFrame()

    wide = (
        df_long.pivot_table(
            index=["actor", "period"],
            columns="group",
            values="ratio",
            aggfunc="sum",
        )
        .reset_index()
        .rename_axis(None, axis=1)
    )

    for col in DEFAULT_GROUPS.keys():
        if col not in wide.columns:
            wide[col] = 0.0

    base = wide["기본관심도"].replace(0, pd.NA)

    wide["로맨스연상_비중"] = wide["로맨스연상"] / base
    wide["연애상대화_비중"] = wide["연애상대화"] / base
    wide["후행탐색_비중"] = wide["후행탐색"] / base
    wide["사생활노이즈_비중"] = wide["사생활노이즈"] / base

    score_cols = [
        "로맨스연상_비중",
        "연애상대화_비중",
        "후행탐색_비중",
        "사생활노이즈_비중",
    ]

    for col in score_cols:
        wide[col] = pd.to_numeric(wide[col], errors="coerce").fillna(0)
        wide[col.replace("_비중", "_백분위")] = (
            wide.groupby("period")[col]
            .rank(pct=True, method="average")
            .mul(100)
            .round(2)
        )

    wide["로맨스잠재력_점수"] = (
        wide["로맨스연상_백분위"] * 0.6
        + wide["연애상대화_백분위"] * 0.4
    ).round(2)

    wide["검색기반_라이징보조점수"] = (
        wide["로맨스잠재력_점수"] * 0.5
        + wide["후행탐색_백분위"] * 0.5
        - wide["사생활노이즈_백분위"] * 0.1
    ).round(2)

    return wide.sort_values(
        ["period", "검색기반_라이징보조점수"],
        ascending=[False, False],
    )


def to_csv_bytes(df: pd.DataFrame) -> bytes:
    return df.to_csv(index=False, encoding="utf-8-sig").encode("utf-8-sig")


def sample_actor_csv() -> bytes:
    sample = pd.DataFrame({"actor": ["김선호", "변우석", "추영우"]})
    return to_csv_bytes(sample)


# =========================
# Streamlit UI
# =========================

st.set_page_config(
    page_title="남배우 라이징 검색지표 수집기",
    page_icon="📈",
    layout="wide",
)

st.title("📈 남배우 라이징 검색지표 수집기")
st.caption("네이버 데이터랩 통합검색어 트렌드 API 기반 · 배우 1명당 1회 호출 · 5개 검색어 그룹")

with st.expander("검색어 그룹 구조", expanded=False):
    st.markdown(
        """
| 그룹 | 검색어 |
|---|---|
| 기본관심도 | 배우명 |
| 로맨스연상 | 배우명 로맨스, 배우명 멜로, 배우명 케미, 배우명 설렘 |
| 연애상대화 | 배우명 남친짤, 배우명 남친미, 배우명 이상형, 배우명 첫사랑 |
| 후행탐색 | 배우명 나무위키, 배우명 인스타 |
| 사생활노이즈 | 배우명 여자친구, 배우명 열애, 배우명 결혼 |
        """
    )

client_id = get_secret("NAVER_CLIENT_ID")
client_secret = get_secret("NAVER_CLIENT_SECRET")

if client_id and client_secret:
    st.success("NAVER_CLIENT_ID / NAVER_CLIENT_SECRET 시크릿이 감지되었습니다.")
else:
    st.error("Streamlit Cloud Secrets에 NAVER_CLIENT_ID / NAVER_CLIENT_SECRET를 먼저 등록하세요.")
    st.code(
        'NAVER_CLIENT_ID = "발급받은_CLIENT_ID"\nNAVER_CLIENT_SECRET = "발급받은_CLIENT_SECRET"',
        language="toml",
    )

st.sidebar.header("수집 설정")

today = date.today()
default_end = today - timedelta(days=1)
default_start = default_end - timedelta(days=27)

start_date = st.sidebar.date_input("시작일", value=default_start)
end_date = st.sidebar.date_input("종료일", value=default_end)

time_unit_label = st.sidebar.selectbox(
    "집계 단위",
    ["date", "week", "month"],
    index=1,
    help="라이징 발굴은 week 추천",
)

device_label = st.sidebar.selectbox("기기", ["전체", "pc", "mo"], index=0)

gender_label = st.sidebar.selectbox(
    "성별",
    ["전체", "m", "f"],
    index=0,
    help="남배우 로맨스 반응을 보려면 f만 별도 수집하는 것도 가능",
)

age_options = {
    "전체": None,
    "10대": ["1"],
    "20대": ["2"],
    "30대": ["3"],
    "40대": ["4"],
    "50대": ["5"],
    "60대 이상": ["6"],
    "10~20대": ["1", "2"],
    "20~30대": ["2", "3"],
    "20~40대": ["2", "3", "4"],
}

age_label = st.sidebar.selectbox("연령", list(age_options.keys()), index=0)

sleep_sec = st.sidebar.number_input(
    "호출 간 대기시간(초)",
    min_value=0.0,
    max_value=3.0,
    value=0.15,
    step=0.05,
)

max_actors = st.sidebar.number_input(
    "이번 실행 최대 배우 수",
    min_value=1,
    max_value=1000,
    value=800,
    step=50,
)

uploaded_file = st.file_uploader(
    "배우 리스트 CSV 업로드",
    type=["csv"],
    help="actor 컬럼이 있으면 actor 컬럼을 사용하고, 없으면 첫 번째 컬럼을 배우명으로 사용합니다.",
)

st.download_button(
    "샘플 CSV 다운로드",
    data=sample_actor_csv(),
    file_name="sample_actors.csv",
    mime="text/csv",
)

actors = []

if uploaded_file is not None:
    try:
        actor_df = pd.read_csv(uploaded_file)
    except UnicodeDecodeError:
        uploaded_file.seek(0)
        actor_df = pd.read_csv(uploaded_file, encoding="cp949")

    if "actor" in actor_df.columns:
        actor_col = "actor"
    else:
        actor_col = actor_df.columns[0]

    actors = (
        actor_df[actor_col]
        .map(normalize_actor_name)
        .loc[lambda s: s.ne("")]
        .drop_duplicates()
        .head(int(max_actors))
        .tolist()
    )

    st.subheader("업로드된 배우 리스트")
    st.write(f"총 {len(actors):,}명")
    st.dataframe(pd.DataFrame({"actor": actors}), use_container_width=True, height=240)

else:
    st.info("CSV를 업로드하면 수집 버튼이 활성화됩니다. CSV에는 actor 컬럼을 두는 것을 추천합니다.")

run = st.button(
    "네이버 데이터랩 수집 시작",
    type="primary",
    disabled=(not actors or not client_id or not client_secret),
)

if run:
    if start_date > end_date:
        st.error("시작일은 종료일보다 늦을 수 없습니다.")
        st.stop()

    st.warning(
        f"이번 실행 예상 API 호출 수: {len(actors):,}회. "
        "네이버 데이터랩 통합검색어 트렌드 API 하루 한도 1,000회를 넘지 않게 관리하세요."
    )

    all_long = []
    errors = []

    progress = st.progress(0)
    status = st.empty()

    for idx, actor in enumerate(actors, start=1):
        status.write(f"[{idx:,}/{len(actors):,}] {actor} 수집 중...")

        try:
            result = call_naver_datalab(
                actor=actor,
                start_date=start_date.strftime("%Y-%m-%d"),
                end_date=end_date.strftime("%Y-%m-%d"),
                time_unit=time_unit_label,
                device=None if device_label == "전체" else device_label,
                gender=None if gender_label == "전체" else gender_label,
                ages=age_options[age_label],
                sleep_sec=float(sleep_sec),
            )
            parsed = parse_api_result(actor, result)
            all_long.append(parsed)

        except Exception as e:
            errors.append({"actor": actor, "error": str(e)})

        progress.progress(idx / len(actors))

    status.write("수집 완료. 결과를 정리합니다.")

    if all_long:
        df_long = pd.concat(all_long, ignore_index=True)
    else:
        df_long = pd.DataFrame(columns=["actor", "period", "group", "ratio"])

    df_summary = make_summary(df_long)
    df_errors = pd.DataFrame(errors)

    st.session_state["df_long"] = df_long
    st.session_state["df_summary"] = df_summary
    st.session_state["df_errors"] = df_errors

    st.success("완료되었습니다.")

if "df_summary" in st.session_state:
    df_summary = st.session_state["df_summary"]
    df_long = st.session_state["df_long"]
    df_errors = st.session_state["df_errors"]

    st.header("결과 요약")

    if not df_summary.empty:
        latest_period = df_summary["period"].max()
        latest = df_summary[df_summary["period"] == latest_period].copy()

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("수집 배우 수", f"{df_summary['actor'].nunique():,}")
        c2.metric("최신 기간", str(latest_period))
        c3.metric("결과 행 수", f"{len(df_summary):,}")
        c4.metric("오류 수", f"{len(df_errors):,}")

        st.subheader("최신 기간 TOP 30")
        top_cols = [
            "actor",
            "period",
            "기본관심도",
            "로맨스잠재력_점수",
            "후행탐색_백분위",
            "사생활노이즈_백분위",
            "검색기반_라이징보조점수",
            "로맨스연상",
            "연애상대화",
            "후행탐색",
            "사생활노이즈",
        ]
        available_cols = [col for col in top_cols if col in latest.columns]
        st.dataframe(
            latest.sort_values("검색기반_라이징보조점수", ascending=False)[available_cols].head(30),
            use_container_width=True,
            height=520,
        )

        st.subheader("전체 Summary")
        st.dataframe(df_summary, use_container_width=True, height=420)

        st.download_button(
            "summary CSV 다운로드",
            data=to_csv_bytes(df_summary),
            file_name="naver_actor_rising_summary.csv",
            mime="text/csv",
        )

        st.download_button(
            "raw long CSV 다운로드",
            data=to_csv_bytes(df_long),
            file_name="naver_actor_rising_raw_long.csv",
            mime="text/csv",
        )

    else:
        st.error("정상 수집된 데이터가 없습니다.")

    if not df_errors.empty:
        st.subheader("오류 목록")
        st.dataframe(df_errors, use_container_width=True)
        st.download_button(
            "errors CSV 다운로드",
            data=to_csv_bytes(df_errors),
            file_name="naver_actor_rising_errors.csv",
            mime="text/csv",
        )
