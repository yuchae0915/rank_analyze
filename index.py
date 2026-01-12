import streamlit as st
import pandas as pd
import requests
import io
import re
import os
import json
import random
from datetime import datetime, timedelta
import plotly.express as px

# 定義 CSS 樣式
hide_icons_css = """
    <style>
    /* 1. 隱藏 "View on GitHub", "Star", "Edit" 等圖示的容器 */
    [data-testid="stHeaderActionElements"] {
        display: none !important;
    }
    
    /* 2. 隱藏 "Share" 按鈕 (有時候它是獨立的元件) */
    .stDeployButton {
        display: none !important;
    }
    
    /* 3. 確保最右邊的 "三點選單" (MainMenu) 依然顯示 */
    [data-testid="stMainMenu"] {
        display: inline-block !important;
    }
    </style>
"""

# 注入 CSS
st.markdown(hide_icons_css, unsafe_allow_html=True)
# ================= 基礎設定 =================
HISTORY_FILE = 'rank_history.json' 
SYSTEM_COLS = ['score', 'dt', 'category', 'threshold_raw', 'threshold_val', 'threshold_col_name', 'Group', 'jitter_y']
MERGE_WINDOW_SECONDS = 120 

# [修改] 移除標題中的圖標
st.set_page_config(page_title="114國營甄試 - 落點分析系統", layout="wide")

# --- 核心函式 ---

def get_default_url():
    try:
        if "general" in st.secrets and "default_url" in st.secrets["general"]:
            return st.secrets["general"]["default_url"]
    except:
        pass
    return ""

def extract_sheet_id(url):
    if not url: return None
    match = re.search(r"/d/([a-zA-Z0-9-_]+)", url)
    return match.group(1) if match else None

def extract_gid(url):
    if not url: return "0"
    match = re.search(r"[#&]gid=([0-9]+)", url)
    return match.group(1) if match else "0"

def build_csv_link(sheet_id, gid):
    return f"https://docs.google.com/spreadsheets/d/{sheet_id}/export?format=csv&gid={gid}"

def clean_score(value):
    try:
        text = str(value)
        match = re.search(r"(\d+(\.\d+)?)", text)
        if match: return float(match.group(1))
        return 0.0
    except:
        return 0.0

def calc_required_interview(opponent_score, my_written, weight_written, weight_interview):
    opponent_interview_assumption = 85.0
    rhs = (opponent_score * weight_written) + (opponent_interview_assumption * weight_interview)
    my_part = (my_written * weight_written)
    if weight_interview == 0: return 999.0
    required_interview = (rhs - my_part) / weight_interview
    return required_interview

def load_history():
    if os.path.exists(HISTORY_FILE):
        try:
            with open(HISTORY_FILE, 'r', encoding='utf-8') as f:
                return json.load(f)
        except:
            return []
    return []

def save_history(history):
    with open(HISTORY_FILE, 'w', encoding='utf-8') as f:
        json.dump(history[-1000:], f, ensure_ascii=False, indent=2)

@st.cache_data(ttl=600)
def fetch_raw_data(target_url):
    try:
        response = requests.get(target_url)
        response.raise_for_status()
        data = pd.read_csv(io.StringIO(response.content.decode('utf-8')))
        
        col_score = [c for c in data.columns if '加權' in c or '成績' in c][0]
        col_cat_list = [c for c in data.columns if '類組' in c]
        col_cat = col_cat_list[0] if col_cat_list else None
        col_threshold = data.columns[-1]

        data['score'] = data[col_score].apply(clean_score)
        data['threshold_val'] = data[col_threshold].apply(clean_score)
        data['threshold_col_name'] = col_threshold 
        data['threshold_raw'] = data[col_threshold] 

        if col_cat:
            data['category'] = data[col_cat].astype(str)
        else:
            data['category'] = "未知類組"

        def parse_time(t_str):
            try:
                t_str = str(t_str).replace('上午', 'AM').replace('下午', 'PM')
                return pd.to_datetime(t_str, format='%Y/%m/%d %p %I:%M:%S', errors='coerce')
            except:
                return pd.Timestamp.now()
        data['dt'] = data['時間戳記'].apply(parse_time)

        return data
    except Exception as e:
        return pd.DataFrame()

# ================= 主程式邏輯 =================

DEFAULT_URL = get_default_url()
query_params = st.query_params
url_sheet_id = query_params.get("id", None)
url_gid = query_params.get("gid", "0")

final_url = None
if url_sheet_id:
    final_url = build_csv_link(url_sheet_id, url_gid)
elif DEFAULT_URL:
    final_url = DEFAULT_URL

# 側邊欄
st.sidebar.header("參數設定控制台")

with st.sidebar.expander("資料來源設定", expanded=not final_url):
    if final_url:
        if url_sheet_id:
            st.success(f"已讀取連結參數 ID: ...{str(url_sheet_id)[-6:]}")
        elif DEFAULT_URL:
            st.success("已讀取本地開發預設值")
        placeholder_text = "貼上新網址以切換表單..."
    else:
        placeholder_text = "請貼上 Google Sheet 網址..."
        
    user_url_input = st.text_input("輸入網址", placeholder=placeholder_text, label_visibility="collapsed")
    st.caption("資料每 10 分鐘自動更新一次。")

if user_url_input:
    new_id = extract_sheet_id(user_url_input)
    new_gid = extract_gid(user_url_input)
    if new_id:
        final_url = build_csv_link(new_id, new_gid)
        st.query_params["id"] = new_id
        st.query_params["gid"] = new_gid
        st.sidebar.success("解析成功！")
        st.sidebar.markdown("---")
        st.sidebar.subheader("分享此設定")
        st.sidebar.info("**請直接複製瀏覽器上方的網址分享**，該網址已包含設定參數。")
    else:
        st.sidebar.error("網址格式錯誤")

if not final_url:
    st.title("114 國營甄試 - 落點分析")
    st.warning("尚未設定資料來源")
    st.markdown("### 快速開始：")
    st.markdown("""
    1. **複製** 您該類組的 Google Sheet 成績表單網址。
    2. **貼上** 到左側選單的輸入框。
    3. **完成！** 系統將自動記憶並分析。
    """)
    st.stop()

raw_data = fetch_raw_data(final_url)
if raw_data.empty:
    st.error("無法連線至資料庫，請檢查網址權限 (需開啟共用) 或網路連線。")
    st.stop()

unique_categories = sorted(raw_data['category'].unique().tolist())
default_index = 0
for i, cat in enumerate(unique_categories):
    if "資訊" in cat:
        default_index = i
        break
selected_category = st.sidebar.selectbox("選擇報考類組", unique_categories, index=default_index)

st.sidebar.subheader("個人數據輸入")
default_score = 50 if "資訊" in selected_category else 0.0
default_quota = 35 if "資訊" in selected_category else 10

my_written_score = st.sidebar.number_input("您的筆試加權成績", value=float(default_score), step=0.1, format="%.2f")
total_quota = st.sidebar.number_input("該類組正取名額", value=int(default_quota), step=1)

is_already_in_list = st.sidebar.checkbox("我的成績已包含在清單中", value=False)

with st.sidebar.expander("進階模型參數"):
    my_interview_worst = st.number_input("我方口試保守預估", value=60.0)
    opponent_interview_best = st.number_input("對手口試極限預估", value=85.0)
    weight_written = 0.8
    weight_interview = 0.2

# 資料處理
df_cat = raw_data[raw_data['category'] == selected_category].copy()

if not df_cat.empty:
    valid_thresholds = df_cat[df_cat['threshold_val'] > 0]['threshold_val']
    if not valid_thresholds.empty:
        pass_threshold = float(valid_thresholds.mode().max())
    else:
        pass_threshold = 0.0
else:
    pass_threshold = 0.0

if pass_threshold > 0:
    df = df_cat[df_cat['score'] >= pass_threshold].copy()
else:
    df = df_cat.copy()

if is_already_in_list and not df.empty:
    matches = df[abs(df['score'] - my_written_score) < 0.001]
    if not matches.empty:
        df = df.drop(matches.index[0])
    else:
        st.sidebar.warning("清單中找不到您的分數。")

df = df.sort_values(by='score', ascending=False).reset_index(drop=True)

# 分析與顯示
if not df.empty:
    competitors = df['score'].tolist()
    threshold_col_name = df['threshold_col_name'].iloc[0]
    
    interview_diff = opponent_interview_best - my_interview_worst
    lead_needed = (interview_diff * weight_interview) / weight_written
    safe_line = my_written_score - lead_needed
    
    raw_rank = sum(s > my_written_score for s in competitors) + 1
    worst_rank = sum(s > safe_line for s in competitors) + 1
    sample_size = len(competitors)
    
    # 歷史紀錄邏輯 - 時間視窗合併
    history = load_history()
    now = datetime.now()
    now_str = now.strftime("%m/%d %H:%M")
    
    save_needed = False
    
    if my_written_score > 0:
        new_record = {
            "time": now_str, 
            "category": selected_category,
            "raw_rank": raw_rank, 
            "worst_rank": worst_rank, 
            "sample_size": sample_size
        }

        if not history:
            history.append(new_record)
            save_history(history)
        else:
            last_rec = history[-1]
            
            if last_rec.get('category') == selected_category:
                try:
                    last_time_struct = datetime.strptime(last_rec['time'], "%m/%d %H:%M")
                    current_time_struct = datetime.strptime(now_str, "%m/%d %H:%M")
                    diff_seconds = (current_time_struct - last_time_struct).total_seconds()
                    
                    if abs(diff_seconds) < MERGE_WINDOW_SECONDS:
                        history[-1] = new_record
                        save_history(history)
                    else:
                        if (raw_rank != last_rec['raw_rank'] or 
                            worst_rank != last_rec['worst_rank'] or 
                            sample_size != last_rec['sample_size']):
                            history.append(new_record)
                            save_history(history)
                except:
                    history.append(new_record)
                    save_history(history)
            else:
                history.append(new_record)
                save_history(history)

    # UI 顯示
    # [修改] 移除標題圖標
    st.title(f"{selected_category} - 落點分析報告")
    
    st.info(f"系統公告：已自動偵測複試門檻為 **{pass_threshold}** 分。系統已自動剔除無效樣本。")
    st.markdown(f"**當前參數**：筆試 `{my_written_score}` | 正取 `{total_quota}` | 來源欄位：`{threshold_col_name}`")

    st.markdown("### 關鍵指標")
    c1, c2, c3, c4 = st.columns(4)
    # [修改] 移除 help 圖標提示
    c1.metric("目前筆試排名", f"No. {raw_rank}")
    c2.metric("最差模擬排名", f"No. {worst_rank}")
    c3.metric("安全分界值", f"{safe_line:.2f} 分")
    c4.metric("有效競爭者 / 總額", f"{sample_size} / {total_quota}")

    # [修改] 移除狀態訊息中的圖標
    if worst_rank <= total_quota:
        st.success(f"**[極度安全]** 模擬最差排名 ({worst_rank}) 仍在正取 ({total_quota}) 內。")
    elif raw_rank <= total_quota:
        st.warning(f"**[需謹慎]** 目前在正取內，但有 {worst_rank - raw_rank} 位對手在射程範圍。")
    else:
        st.error(f"**[危險]** 目前排名在正取外，需靠口試高分逆轉。")

    st.divider()

    # [新的] 競爭與逆轉分析 (移除圖標版)
    st.subheader("競爭與逆轉分析")
    
    # === 情境 A：我在正取名單內 (防守模式) ===
    if raw_rank <= total_quota:
        st.success(f"目前排名 **No.{raw_rank}** (正取 {total_quota})，處於安全名單內！")
        
        threats = df[(df['score'] > safe_line) & (df['score'] < my_written_score)].copy()
        
        if not threats.empty:
            st.markdown("##### 需警戒的後方對手")
            st.caption("這些對手筆試輸您，但若口試表現優異，可能總分會超越您。")
            
            threats['筆試落後'] = (my_written_score - threats['score']).round(2)
            threats['口試需贏我'] = (threats['筆試落後'] * (weight_written / weight_interview)).round(2)
            
            display_cols = ['加權成績', '筆試落後', '口試需贏我']
            st.dataframe(threats[display_cols].sort_values('加權成績', ascending=False).reset_index(drop=True), use_container_width=True)
        else:
            st.markdown("##### 防守狀況：極度安全")
            st.info("目前後方無人在「射程範圍」內。除非您口試失常（低於 60）且對手滿分， otherwise 您幾乎確定上榜。")

    # === 情境 B：我在正取名單外 (進攻模式) ===
    else:
        diff_rank = raw_rank - total_quota
        st.error(f"目前排名 **No.{raw_rank}** (正取 {total_quota})，暫時落後 **{diff_rank}** 名。")
        
        if len(competitors) >= total_quota:
            cutoff_score = competitors[total_quota - 1]
        else:
            cutoff_score = competitors[-1]
            
        targets = df[(df['score'] > my_written_score) & (df['score'] >= cutoff_score)].copy()
        
        if not targets.empty:
            st.markdown("##### 逆轉勝策略分析")
            st.caption(f"筆試輸 1 分，口試需贏 4 分。以下是您必須擊敗的對手門檻：")
            
            targets['筆試領先'] = (targets['score'] - my_written_score).round(2)
            targets['口試需贏'] = (targets['筆試領先'] * (weight_written / weight_interview)).round(2)
            
            def judge_difficulty(lead_needed):
                if lead_needed <= 5: return "[易] (贏 5 分內)"
                if lead_needed <= 10: return "[中] (贏 5-10 分)"
                if lead_needed <= 15: return "[難] (贏 10-15 分)"
                return "[極難] (需贏 >15 分)"

            targets['逆轉難度'] = targets['口試需贏'].apply(judge_difficulty)
            
            display_cols = ['加權成績', '筆試領先', '口試需贏', '逆轉難度']
            display_targets = targets[display_cols].sort_values('加權成績', ascending=True).head(10).reset_index(drop=True)
            
            st.dataframe(display_targets, use_container_width=True)
            
            min_catchup = display_targets.iloc[0]['口試需贏']
            
            st.markdown(f"""
            **分析結論：**
            * 您至少需要追過 **{diff_rank}** 個人才能擠進正取。
            * 距離您最近的對手（正取尾），筆試贏您 `{display_targets.iloc[0]['筆試領先']}` 分。
            * **您的口試成績必須比對方高出 `{min_catchup}` 分** 才能逆轉勝。
            """)
            
            if min_catchup > 20:
                st.warning("逆轉所需的口試分差超過 20 分，翻盤難度極高，需祈禱對手口試嚴重失常。")
        else:
            st.info("前方資料不足，無法計算逆轉所需分數。")

    st.divider()

    st.subheader("有效競爭者分布")
    def categorize(score):
        if score == my_written_score: return "Self (我方)" 
        if score > my_written_score: return "Leading (領先群)"
        if score > safe_line: return "Competitors (競爭區間)"
        return "Safe (安全區間)"

    df['Group'] = df['score'].apply(categorize)
    df['jitter_y'] = [random.uniform(0, 1) for _ in range(len(df))]
    
    x_min = pass_threshold - 0.5
    x_max = df['score'].max() + 1 if not df.empty else 100

    fig_dist = px.scatter(df, x="score", y="jitter_y", color="Group", 
                        hover_data=["時間戳記", "加權成績"],
                        color_discrete_map={
                            "Self (我方)": "#D62728", "Leading (領先群)": "#7F7F7F",  
                            "Competitors (競爭區間)": "#FF7F0E", "Safe (安全區間)": "#2CA02C"    
                        })
    fig_dist.update_traces(marker=dict(size=10, opacity=0.9))
    fig_dist.update_layout(
        height=280,
        xaxis=dict(title="筆試加權成績", range=[x_min, x_max]),
        yaxis_visible=False,
        margin=dict(l=20, r=20, t=30, b=20),
        legend=dict(title="群組分類", orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    fig_dist.add_vline(x=my_written_score, line_dash="dash", line_width=1, line_color="#D62728")
    fig_dist.add_vline(x=safe_line, line_dash="dash", line_width=1, line_color="#2CA02C")
    fig_dist.add_vline(x=pass_threshold, line_dash="dot", line_width=1, line_color="black")
    st.plotly_chart(fig_dist, use_container_width=True)

    st.subheader("排名趨勢 (當前類組)")
    df_hist = pd.DataFrame(history)
    if not df_hist.empty and 'category' in df_hist.columns:
        df_hist_filtered = df_hist[df_hist['category'] == selected_category].copy()
    else:
        df_hist_filtered = pd.DataFrame()

    if not df_hist_filtered.empty:
        fig = px.line(df_hist_filtered, x='time', y=['worst_rank', 'raw_rank'], markers=True)
        y_max = df_hist_filtered['worst_rank'].max() + 2
        fig.update_layout(
            yaxis=dict(range=[y_max, 0.5], title="排名", dtick=5), 
            xaxis=dict(title="時間"),
            height=400,
            legend=dict(title="指標", orientation="h", y=1.1, x=1)
        )
        fig.add_hline(y=total_quota, line_dash="dash", line_width=1, line_color="red", annotation_text="正取線")
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("此類組尚無歷史紀錄。")

    with st.expander("原始資料檢視"):
        tab1, tab2 = st.tabs(["有效名單 (已過濾)", "全部資料"])
        
        def show_clean_dataframe(dataframe):
            clean_df = dataframe.drop(columns=SYSTEM_COLS, errors='ignore').copy()
            for col in clean_df.columns:
                if clean_df[col].dtype in ['float64', 'int64']:
                    clean_df[col] = clean_df[col].astype(str)
            clean_df.reset_index(drop=True, inplace=True)
            clean_df.index += 1
            st.dataframe(clean_df, use_container_width=True)

        with tab1:
            show_clean_dataframe(df)
        with tab2:
            show_clean_dataframe(raw_data)

else:
    st.warning(f"目前類組 `{selected_category}` 尚無有效數據。")

st.sidebar.markdown("---")
st.sidebar.subheader("危險操作區")
if st.sidebar.button("清除本機歷史數據", type="primary"):
    if os.path.exists(HISTORY_FILE):
        os.remove(HISTORY_FILE)
        st.rerun()