import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from pathlib import Path

st.set_page_config(
    page_title="Digital Payment Fraud Shield",
    page_icon="🛡️",
    layout="wide"
)

st.title("🛡️ Digital Payment Fraud Shield")
st.caption("Dashboard giám sát rủi ro giao dịch thanh toán số — công cụ hỗ trợ rà soát, không kết luận gian lận.")

REASON_COLUMNS = {
    "flag_amount_spike": "Số tiền tăng bất thường",
    "flag_high_txn_velocity": "Tần suất giao dịch cao",
    "flag_high_txn_count_24h": "Nhiều giao dịch trong 24 giờ",
    "flag_location_change": "Thay đổi địa điểm",
    "flag_channel_mismatch": "Khác kênh thanh toán ưa dùng",
    "flag_registered_device_mismatch": "Thiết bị không khớp đăng ký",
    "registered_customer_mismatch": "Thiết bị không khớp khách hàng",
    "location_changed_vs_prev": "Thay đổi địa điểm so với trước",
    "channel_mismatch_preferred": "Không khớp kênh ưa dùng",
    "emulator_flag": "Thiết bị giả lập",
    "rooted_or_jailbroken": "Thiết bị đã root/jailbreak",
}

@st.cache_data
def read_file(file_bytes, filename):
    suffix = Path(filename).suffix.lower()
    from io import BytesIO
    bio = BytesIO(file_bytes)
    if suffix == ".csv":
        try:
            return pd.read_csv(bio)
        except UnicodeDecodeError:
            bio.seek(0)
            return pd.read_csv(bio, encoding="latin-1")
    if suffix in [".xlsx", ".xls"]:
        return pd.read_excel(bio)
    raise ValueError("Chỉ hỗ trợ tệp CSV hoặc Excel.")

def to_numeric_safe(series):
    if pd.api.types.is_numeric_dtype(series):
        return pd.to_numeric(series, errors="coerce")
    s = series.astype("string").str.strip()
    # Decimal comma is converted only when no decimal point is present.
    comma_decimal = s.str.contains(",", na=False) & ~s.str.contains(r"\.", na=False)
    s.loc[comma_decimal] = s.loc[comma_decimal].str.replace(",", ".", regex=False)
    return pd.to_numeric(s, errors="coerce")

def make_prototype_scores(frame):
    """Fallback only: transparent rule-based prototype, not a trained ML model."""
    d = frame.copy()
    rule_map = {
        "flag_amount_spike": 1.0,
        "flag_high_txn_velocity": 1.0,
        "flag_high_txn_count_24h": 1.0,
        "flag_location_change": 1.0,
        "flag_channel_mismatch": 1.0,
        "flag_registered_device_mismatch": 1.0,
        "registered_customer_mismatch": 1.0,
    }
    present = [c for c in rule_map if c in d.columns]
    if present:
        flags = pd.DataFrame(index=d.index)
        for c in present:
            flags[c] = to_numeric_safe(d[c]).fillna(0).clip(0, 1)
        d["risk_score"] = flags.mean(axis=1).fillna(0)
    else:
        d["risk_score"] = 0.0
    d["risk_level"] = pd.cut(
        d["risk_score"],
        bins=[-np.inf, 0.20, 0.50, 0.80, np.inf],
        labels=["LOW", "MEDIUM", "HIGH", "CRITICAL"],
        right=False
    ).astype(str)
    d["predicted_fraud"] = (d["risk_score"] >= 0.5).astype(int)
    d["model_name"] = "Rule-based prototype (not ML)"
    d["alert_reason"] = ""
    for idx, row in d.iterrows():
        reasons = []
        for col in present:
            if pd.notna(row[col]) and to_numeric_safe(pd.Series([row[col]])).iloc[0] >= 1:
                reasons.append(REASON_COLUMNS.get(col, col))
        d.at[idx, "alert_reason"] = "; ".join(reasons) if reasons else "Không kích hoạt rule trong prototype"
    return d, bool(present)

with st.sidebar:
    st.header("Dữ liệu")
    uploaded = st.file_uploader("Tải file kết quả CSV/Excel", type=["csv", "xlsx", "xls"])
    st.caption("Ưu tiên dùng digital_payment_fraud_shield_output.csv hoặc .xlsx sau khi chạy pipeline ML.")
    st.divider()
    st.header("Bộ lọc")
    st.caption("Các bộ lọc xuất hiện sau khi dữ liệu được nạp.")

if uploaded is not None:
    try:
        df = read_file(uploaded.getvalue(), uploaded.name)
        source_name = uploaded.name
    except Exception as e:
        st.error(f"Không đọc được file: {e}")
        st.stop()
else:
    candidates = [
        Path("digital_payment_fraud_shield_output.csv"),
        Path("digital_payment_fraud_shield_output.xlsx"),
        Path("labeled_transaction_behavioral_feature.xlsx"),
        Path("/mnt/data/labeled_transaction_behavioral_feature.xlsx"),
    ]
    chosen = next((p for p in candidates if p.exists()), None)
    if chosen is None:
        st.info("Hãy tải lên file kết quả CSV/Excel bằng thanh bên trái để bắt đầu.")
        st.stop()
    try:
        df = pd.read_csv(chosen) if chosen.suffix == ".csv" else pd.read_excel(chosen)
        source_name = str(chosen)
    except Exception as e:
        st.error(f"Không đọc được file {chosen}: {e}")
        st.stop()

df.columns = [str(c).strip() for c in df.columns]
if df.empty:
    st.error("File không có dòng dữ liệu.")
    st.stop()

# If scores are absent, show an explicitly labelled rule-based prototype.
prototype_mode = False
if "risk_score" not in df.columns:
    df, has_rules = make_prototype_scores(df)
    prototype_mode = True
    st.warning(
        "ĐANG Ở CHẾ ĐỘ PROTOTYPE: file chưa có risk_score do mô hình tạo ra. "
        "Dashboard đang dùng điểm rule minh bạch để minh họa, không phải dự đoán ML đã kiểm chứng."
    )
else:
    df["risk_score"] = to_numeric_safe(df["risk_score"]).clip(0, 1)

if "risk_level" not in df.columns:
    df["risk_level"] = pd.cut(
        df["risk_score"].fillna(0),
        bins=[-np.inf, 0.20, 0.50, 0.80, np.inf],
        labels=["LOW", "MEDIUM", "HIGH", "CRITICAL"],
        right=False
    ).astype(str)
if "predicted_fraud" not in df.columns:
    df["predicted_fraud"] = (df["risk_score"].fillna(0) >= 0.5).astype(int)
if "priority_rank" not in df.columns:
    df["priority_rank"] = df["risk_score"].rank(method="first", ascending=False).astype(int)
if "alert_reason" not in df.columns:
    df["alert_reason"] = "Chưa có alert_reason trong file đầu vào"
if "transaction_id" not in df.columns:
    df["transaction_id"] = [f"ROW-{i+1:06d}" for i in range(len(df))]

# Normalize labels for display, without interpreting missing labels as normal.
if "is_fraud" in df.columns:
    df["is_fraud_numeric"] = pd.to_numeric(df["is_fraud"], errors="coerce")
else:
    df["is_fraud_numeric"] = np.nan

st.caption(f"Nguồn dữ liệu: `{source_name}` · {len(df):,} dòng · {len(df.columns):,} cột")

# Sidebar filters
with st.sidebar:
    levels = ["CRITICAL", "HIGH", "MEDIUM", "LOW"]
    available_levels = [x for x in levels if x in set(df["risk_level"].astype(str))]
    selected_levels = st.multiselect("Mức cảnh báo", available_levels, default=available_levels)
    search_id = st.text_input("Tìm transaction ID")
    if "payment_channel" in df.columns:
        channels = sorted(df["payment_channel"].dropna().astype(str).unique().tolist())
        selected_channels = st.multiselect("Kênh thanh toán", channels, default=channels)
    else:
        selected_channels = None

filtered = df[df["risk_level"].astype(str).isin(selected_levels)].copy()
if search_id.strip():
    filtered = filtered[filtered["transaction_id"].astype(str).str.contains(search_id.strip(), case=False, na=False)]
if selected_channels is not None:
    filtered = filtered[filtered["payment_channel"].astype(str).isin(selected_channels)]

# KPI cards
n = len(df)
n_alert = int(df["risk_level"].astype(str).isin(["HIGH", "CRITICAL"]).sum())
n_review = int(df["risk_level"].astype(str).isin(["MEDIUM", "HIGH", "CRITICAL"]).sum())
labeled_mask = df["is_fraud_numeric"].isin([0, 1])
n_labeled = int(labeled_mask.sum())
k1, k2, k3, k4 = st.columns(4)
k1.metric("Tổng giao dịch", f"{n:,}")
k2.metric("HIGH / CRITICAL", f"{n_alert:,}", f"{n_alert / max(n,1):.1%} tổng giao dịch")
k3.metric("Cần rà soát (MEDIUM+)", f"{n_review:,}")
k4.metric("Giao dịch có nhãn", f"{n_labeled:,}")

tab1, tab2, tab3 = st.tabs(["Tổng quan", "Danh sách giao dịch", "Chi tiết & đánh giá"])

with tab1:
    left, right = st.columns(2)
    with left:
        level_counts = df["risk_level"].astype(str).value_counts().reindex(
            ["LOW", "MEDIUM", "HIGH", "CRITICAL"], fill_value=0
        ).rename_axis("risk_level").reset_index(name="count")
        fig = px.bar(level_counts, x="risk_level", y="count", title="Phân bố mức cảnh báo")
        st.plotly_chart(fig, use_container_width=True)
    with right:
        if "payment_channel" in df.columns:
            channel_counts = df["payment_channel"].fillna("Unknown").astype(str).value_counts().head(12).rename_axis("payment_channel").reset_index(name="count")
            fig2 = px.bar(channel_counts, x="payment_channel", y="count", title="Giao dịch theo kênh thanh toán")
            st.plotly_chart(fig2, use_container_width=True)
        else:
            fig2 = px.histogram(df, x="risk_score", nbins=20, title="Phân bố risk score")
            st.plotly_chart(fig2, use_container_width=True)

    fig3 = px.histogram(df, x="risk_score", color="risk_level", nbins=30, title="Phân bố điểm rủi ro")
    st.plotly_chart(fig3, use_container_width=True)

    if "datetime" in df.columns:
        dt = pd.to_datetime(df["datetime"], errors="coerce")
        time_df = pd.DataFrame({"datetime": dt, "risk_score": df["risk_score"], "risk_level": df["risk_level"]}).dropna(subset=["datetime"])
        if not time_df.empty:
            time_df["date"] = time_df["datetime"].dt.date
            daily = time_df.groupby("date").agg(
                transactions=("risk_score", "size"),
                mean_risk=("risk_score", "mean"),
                high_critical=("risk_level", lambda s: s.astype(str).isin(["HIGH", "CRITICAL"]).sum())
            ).reset_index()
            fig4 = px.line(daily, x="date", y="high_critical", markers=True, title="Số cảnh báo HIGH/CRITICAL theo ngày")
            st.plotly_chart(fig4, use_container_width=True)

with tab2:
    st.subheader("Danh sách giao dịch theo mức ưu tiên")
    sort_col = "risk_score"
    view = filtered.sort_values(sort_col, ascending=False)
    preferred = [
        "priority_rank", "transaction_id", "datetime", "amount", "currency",
        "payment_channel", "merchant_name", "transaction_status",
        "risk_score", "risk_level", "predicted_fraud", "alert_reason"
    ]
    shown_cols = [c for c in preferred if c in view.columns]
    if not shown_cols:
        shown_cols = view.columns.tolist()[:12]
    st.dataframe(view[shown_cols], use_container_width=True, hide_index=True)
    csv_bytes = view.to_csv(index=False).encode("utf-8-sig")
    st.download_button("⬇️ Tải danh sách đang lọc (CSV)", csv_bytes, "filtered_transactions.csv", "text/csv")

with tab3:
    st.subheader("Tra cứu một giao dịch")
    ids = df["transaction_id"].astype(str).tolist()
    selected_id = st.selectbox("Chọn transaction_id", ids)
    row = df[df["transaction_id"].astype(str) == selected_id].iloc[0]
    a, b, c = st.columns(3)
    a.metric("Risk score", f"{float(row['risk_score']):.3f}" if pd.notna(row["risk_score"]) else "N/A")
    b.metric("Risk level", str(row["risk_level"]))
    c.metric("Priority rank", f"{int(row['priority_rank'])}" if pd.notna(row["priority_rank"]) else "N/A")
    st.write("**Lý do cảnh báo / bằng chứng**")
    st.info(str(row.get("alert_reason", "Chưa có lý do cảnh báo.")))

    evidence = []
    for col, label in REASON_COLUMNS.items():
        if col in df.columns:
            value = row[col]
            if pd.notna(value) and str(value).strip().lower() not in ["0", "0.0", "false", "no", "nan", ""]:
                evidence.append({"Tín hiệu": label, "Cột dữ liệu": col, "Giá trị": str(value)})
    if evidence:
        st.dataframe(pd.DataFrame(evidence), use_container_width=True, hide_index=True)
    else:
        st.caption("Không tìm thấy cờ rule hoạt động trong các cột tín hiệu đã biết.")

    st.write("**Thông tin giao dịch**")
    detail_cols = [
        "transaction_id", "datetime", "amount", "currency", "payment_channel",
        "transaction_status", "customer_id", "merchant_id", "merchant_name",
        "merchant_category", "location", "ip_country", "device_id",
        "authentication_method", "risk_score", "risk_level", "alert_reason"
    ]
    detail_cols = [c for c in detail_cols if c in df.columns]
    st.dataframe(row[detail_cols].rename("Giá trị").to_frame(), use_container_width=True)

    if n_labeled:
        st.divider()
        st.write("**Đánh giá trên giao dịch có nhãn**")
        eval_df = df.loc[labeled_mask].copy()
        st.caption("Chỉ các dòng có is_fraud = 0 hoặc 1 được dùng. Dòng chưa có nhãn không được coi là bình thường.")
        if "predicted_fraud" in eval_df.columns:
            from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix
            # Prefer a dedicated held-out test subset if the file includes a split column.
            if "split" in eval_df.columns:
                eval_df["split_norm"] = eval_df["split"].astype(str).str.lower()
                test_eval = eval_df[eval_df["split_norm"].isin(["test", "testing"])]
            else:
                test_eval = pd.DataFrame()

            if not test_eval.empty:
                st.success("KPI dưới đây được tính trên các dòng được đánh dấu split = test.")
                metric_df = test_eval
            else:
                st.warning(
                    "File chưa có cột split=test. Các chỉ số dưới đây (nếu xem) chỉ là thống kê trên "
                    "các dòng có nhãn trong file, KHÔNG phải KPI test độc lập. "
                    "Không dùng chúng làm kết quả chính thức trong hồ sơ dự thi."
                )
                metric_df = eval_df

            yt = metric_df["is_fraud_numeric"].astype(int)
            yp = pd.to_numeric(metric_df["predicted_fraud"], errors="coerce").fillna(0).astype(int)
            p, r, f = st.columns(3)
            p.metric("Precision", f"{precision_score(yt, yp, zero_division=0):.3f}")
            r.metric("Recall", f"{recall_score(yt, yp, zero_division=0):.3f}")
            f.metric("F1-score", f"{f1_score(yt, yp, zero_division=0):.3f}")
            st.write("Confusion matrix (thứ tự nhãn 0, 1):")
            cm = confusion_matrix(yt, yp, labels=[0,1])
            st.dataframe(pd.DataFrame(cm, index=["Thực tế 0", "Thực tế 1"], columns=["Dự đoán 0", "Dự đoán 1"]))
        else:
            st.caption("Chưa có cột predicted_fraud để tính KPI.")

st.divider()
st.caption(
    "Lưu ý: cảnh báo chỉ hỗ trợ sàng lọc và ưu tiên rà soát. Không xem một cảnh báo là kết luận gian lận. "
    + ("Điểm hiện tại là prototype dựa trên rule, chưa phải mô hình ML đã kiểm chứng." if prototype_mode else "KPI chỉ có ý nghĩa trên dữ liệu có nhãn và tập đánh giá được xác định rõ.")
)
