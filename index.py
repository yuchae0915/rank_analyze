import streamlit as st
import pandas as pd
import requests
import io
import re
import os
import json
import random
from datetime import datetime, timedelta # [新增] timedelta
import plotly.express as px

# ================= 基礎設定 =================
HISTORY_FILE = 'rank_history.json' 
SYSTEM_COLS = ['score', 'dt', 'category', 'threshold_raw', 'threshold_val', 'threshold_col_name', 'Group', 'jitter_y']

# [新增] 定義合併時間視窗 (秒)，在此時間內的連續操作會被覆蓋，不會產生新點
MERGE_WINDOW_SECONDS = 120 

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
    """
    [修改] 這裡只負責單純寫入，邏輯判斷移到主程式
    """
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
    st.caption("💡 資料每 10 分鐘自動更新一次。")

if user_url_input:
    new_id = extract_sheet_id(user_url_input)
    new_gid = extract_gid(user_url_input)
    if new_id:
        final_url = build_csv_link(new_id, new_gid)
        st.query_params["id"] = new_id
        st.query_params["gid"] = new_gid
        st.sidebar.success("✅ 解析成功！")
        st.sidebar.markdown("---")
        st.sidebar.subheader("🔗 分享此設定")
        st.sidebar.info("💡 **請直接複製瀏覽器上方的網址分享**，該網址已包含設定參數。")
    else:
        st.sidebar.error("❌ 網址格式錯誤")

if not final_url:
    st.title("📊 114 國營甄試落點分析")
    st.warning("⚠️ 尚未設定資料來源")
    st.markdown("### 🚀 快速開始：")
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
default_score = 57.4 if "資訊" in selected_category else 0.0
default_quota = 35 if "資訊" in selected_category else 10
my_written_score = st.sidebar.number_input("您的筆試加權成績", value=default_score, step=0.1, format="%.2f")
total_quota = st.sidebar.number_input("該類組正取名額", value=default_quota, step=1)

is_already_in_list = st.sidebar.checkbox("我的成績已包含在清單中", value=False, help="系統將自動排除一筆與您同分的資料。")

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
    
    # [修改] 歷史紀錄邏輯 - 時間視窗合併
    history = load_history()
    now = datetime.now()
    now_str = now.strftime("%m/%d %H:%M") # 用於顯示的字串
    
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
            
            # 1. 檢查是否同類組
            if last_rec.get('category') == selected_category:
                # 2. 檢查時間差 (解析上一筆時間)
                try:
                    # 這裡將字串轉回 datetime，注意年份會預設為 1900，所以我們把現在時間也轉成 1900 來比較
                    last_time_struct = datetime.strptime(last_rec['time'], "%m/%d %H:%M")
                    current_time_struct = datetime.strptime(now_str, "%m/%d %H:%M")
                    
                    # 計算秒數差
                    diff_seconds = (current_time_struct - last_time_struct).total_seconds()
                    
                    # [關鍵] 若在時間視窗內 (例如 10 分鐘)
                    if abs(diff_seconds) < MERGE_WINDOW_SECONDS:
                        # 覆蓋上一筆 (Update)
                        history[-1] = new_record
                        save_history(history)
                    else:
                        # 超過時間，新增一筆 (Append)
                        # 只有當數據有變化時才存，避免長時間掛機產生大量重複數據
                        if (raw_rank != last_rec['raw_rank'] or 
                            worst_rank != last_rec['worst_rank'] or 
                            sample_size != last_rec['sample_size']):
                            history.append(new_record)
                            save_history(history)
                except:
                    # 如果時間解析失敗，就直接存新的
                    history.append(new_record)
                    save_history(history)
            else:
                # 不同類組，直接存新的
                history.append(new_record)
                save_history(history)

    # UI 顯示
    st.title(f"{selected_category} - 落點分析報告")
    
    st.info(f"系統公告：已自動偵測複試門檻為 **{pass_threshold}** 分。系統已自動剔除無效樣本。")
    st.markdown(f"**當前參數**：筆試 `{my_written_score}` | 正取 `{total_quota}` | 來源欄位：`{threshold_col_name}`")

    st.markdown("### 關鍵指標")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("目前筆試排名", f"No. {raw_rank}")
    c2.metric("最差模擬排名", f"No. {worst_rank}", help="保守估計排名")
    c3.metric("安全分界值", f"{safe_line:.2f} 分")
    
    sample_help = "已排除您自身資料 (參賽者模式)" if is_already_in_list else "包含所有填表資料 (觀察者模式)"
    c4.metric("有效競爭者 / 總額", f"{sample_size} / {total_quota}", help=sample_help)

    if worst_rank <= total_quota:
        st.success(f"**[極度安全]** 模擬最差排名 ({worst_rank}) 仍在正取 ({total_quota}) 內。")
    elif raw_rank <= total_quota:
        st.warning(f"**[需謹慎]** 目前在正取內，但有 {worst_rank - raw_rank} 位對手在射程範圍。")
    else:
        st.error(f"**[危險]** 目前排名在正取外，需靠口試高分逆轉。")

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

    st.subheader("競爭區間對手分析")
    threats = df[(df['score'] > safe_line) & (df['score'] < my_written_score)].copy()
    if not threats.empty:
        threats['分差'] = (my_written_score - threats['score']).round(2)
        def get_win_strategy(row):
            req = calc_required_interview(row['score'], my_written_score, weight_written, weight_interview)
            if req > 100: return "無法超越"
            if req <= 60: return "60 (及格即勝)"
            return f"{req:.2f}"
        threats['所需口試分數'] = threats.apply(get_win_strategy, axis=1)
        
        display_threats = threats[['加權成績', '分差', '所需口試分數']].sort_values('加權成績', ascending=False).reset_index(drop=True)
        for col in ['加權成績', '分差', '所需口試分數']:
            display_threats[col] = display_threats[col].astype(str)
        display_threats.index += 1
        st.dataframe(display_threats, use_container_width=True)
    else:
        st.info("目前無人位於競爭區間 (安全)。")

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