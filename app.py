import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from FinMind.data import DataLoader
from datetime import datetime, timedelta
import json
import io
from fpdf import FPDF
import tempfile
import os
import mplfinance as mpf

# --- 設定頁面資訊 ---
st.set_page_config(page_title="Gemini 股市戰情室", page_icon="📈", layout="wide")

# --- 初始化 Session State (記憶體) ---
if 'analyzed_data' not in st.session_state:
    st.session_state['analyzed_data'] = None
if 'pdf_bytes' not in st.session_state:
    st.session_state['pdf_bytes'] = None
if 'json_txt' not in st.session_state:
    st.session_state['json_txt'] = None

# --- 核心邏輯函數 ---
@st.cache_data(ttl=3600)
def get_stock_data(tickers):
    try:
        # 下載包含 Open, High, Low, Close, Volume 的完整數據
        # group_by='ticker' 確保多檔股票格式統一
        df = yf.download(tickers, period="1y", progress=False, auto_adjust=True, group_by='ticker')
        return df
    except Exception as e:
        st.error(f"數據下載失敗: {e}")
        return pd.DataFrame()

def calculate_kd(close, high, low, n=9):
    try:
        close = pd.Series(close)
        high = pd.Series(high)
        low = pd.Series(low)
        low_min = low.rolling(window=n).min()
        high_max = high.rolling(window=n).max()
        rsv = (close - low_min) / (high_max - low_min) * 100
        rsv = rsv.fillna(50)
        k_values, d_values = [50], [50]
        for i in range(1, len(rsv)):
            k = (2/3) * k_values[-1] + (1/3) * rsv.iloc[i]
            d = (2/3) * d_values[-1] + (1/3) * k
            k_values.append(k)
            d_values.append(d)
        return k_values[-1], d_values[-1], k_values[-2], d_values[-2]
    except:
        return 50, 50, 50, 50

def get_finmind_data(stock_id):
    # FinMind 不需要 .TW 或 .TWO，只需要純數字代號
    clean_id = stock_id.replace('.TW', '').replace('.TWO', '')
    fm = DataLoader()
    
    # 1. 籌碼
    df_chips = fm.taiwan_stock_institutional_investors(
        stock_id=clean_id,
        start_date=(datetime.now() - timedelta(days=40)).strftime('%Y-%m-%d'),
        end_date=datetime.now().strftime('%Y-%m-%d')
    )
    
    foreign_buy, trust_buy = 0, 0
    if not df_chips.empty:
        f_data = df_chips[df_chips['name'] == 'Foreign_Investor']
        t_data = df_chips[df_chips['name'] == 'Investment_Trust']
        if not f_data.empty:
            foreign_buy = ((f_data['buy'] - f_data['sell']) / 1000).tail(5).sum()
        if not t_data.empty:
            trust_buy = ((t_data['buy'] - t_data['sell']) / 1000).tail(5).sum()

    # 2. 營收
    df_rev = fm.taiwan_stock_month_revenue(
        stock_id=clean_id,
        start_date=(datetime.now() - timedelta(days=450)).strftime('%Y-%m-%d'),
        end_date=datetime.now().strftime('%Y-%m-%d')
    )
    
    rev_yoy = None
    current_rev = 0
    rev_msg = "N/A"
    
    if not df_rev.empty:
        df_rev['revenue'] = pd.to_numeric(df_rev['revenue'], errors='coerce')
        latest = df_rev.iloc[-1]
        current_rev = latest['revenue']
        this_year, this_month = int(latest['revenue_year']), int(latest['revenue_month'])
        
        prev = df_rev[(df_rev['revenue_year'] == this_year - 1) & (df_rev['revenue_month'] == this_month)]
        if not prev.empty:
            last_rev = prev.iloc[0]['revenue']
            if last_rev > 0:
                rev_yoy = ((current_rev - last_rev) / last_rev) * 100
                rev_msg = f"{rev_yoy:.2f}%"
            else:
                rev_msg = "No Base"
        else:
            rev_msg = "No Data"

    return foreign_buy, trust_buy, rev_yoy, rev_msg, current_rev

# --- PDF 生成類別 ---
class ReportPDF(FPDF):
    def header(self):
        self.set_font('Arial', 'B', 15)
        self.cell(0, 10, 'Gemini Stock Analysis Report', 0, 1, 'C')
        self.ln(10)

    def footer(self):
        self.set_y(-15)
        self.set_font('Arial', 'I', 8)
        self.cell(0, 10, f'Page {self.page_no()}', 0, 0, 'C')

# --- UI 介面設計 ---

st.title("📊 Gemini 投資戰情室")
st.markdown("結合 **技術面 (KD/均線)** + **籌碼面 (外資/投信)** + **基本面 (營收YoY)** 的三合一分析工具")

with st.sidebar:
    st.header("設定")
    # 預設值加入一些上櫃股票測試
    default_stocks = "2330, 6873, 6488"
    user_input = st.text_area("輸入股票代號 (用逗號分隔)", default_stocks, height=100)
    
    if st.button("🚀 開始分析", type="primary"):
        # 1. 整理使用者輸入的代號
        raw_symbols = [s.strip().upper() for s in user_input.split(',') if s.strip()]
        
        # 2. 【關鍵修正】為每個代號同時產生 .TW (上市) 和 .TWO (上櫃) 兩種可能
        search_list = []
        for s in raw_symbols:
            # 如果使用者自己沒打 .TW 或 .TWO，我們就幫他兩個都猜
            if '.TW' not in s and '.TWO' not in s:
                search_list.append(f'{s}.TW')
                search_list.append(f'{s}.TWO')
            else:
                search_list.append(s) # 使用者自己有打後綴就照用
        
        with st.spinner(f"正在掃描 {len(raw_symbols)} 檔股票 (嘗試上市/上櫃匹配)..."):
            # 3. 一次性下載所有可能的代號
            df_all = get_stock_data(search_list)
            
            processed_data = [] 
            all_ai_data_list = []
            pdf = ReportPDF()
            pdf.set_auto_page_break(auto=True, margin=15)
            
            # 4. 逐一檢查哪個代號才是真的
            for stock_code in raw_symbols:
                valid_ticker = None
                stock_df = None
                
                # 嘗試找 .TW
                try:
                    if f"{stock_code}.TW" in df_all.columns.levels[0]: # 檢查第一層索引(Ticker)
                        temp_df = df_all[f"{stock_code}.TW"]
                        # 檢查是不是全是空值 (有些下市股會有欄位但沒數據)
                        if not temp_df.isnull().all().all():
                            valid_ticker = f"{stock_code}.TW"
                            stock_df = temp_df
                except: pass
                
                # 如果 .TW 沒資料，嘗試找 .TWO
                if valid_ticker is None:
                    try:
                        if f"{stock_code}.TWO" in df_all.columns.levels[0]:
                            temp_df = df_all[f"{stock_code}.TWO"]
                            if not temp_df.isnull().all().all():
                                valid_ticker = f"{stock_code}.TWO"
                                stock_df = temp_df
                    except: pass

                # 如果兩個都找不到，那就真的是無效代號
                if valid_ticker is None or stock_df is None:
                    st.warning(f"⚠️ 找不到 {stock_code} 的數據 (可能代號錯誤或已下市)")
                    continue

                # --- 接下來的邏輯與之前相同，使用 valid_ticker 繼續處理 ---
                
                try:
                    ohlc_data = pd.DataFrame({
                        'Open': stock_df['Open'],
                        'High': stock_df['High'],
                        'Low': stock_df['Low'],
                        'Close': stock_df['Close'],
                        'Volume': stock_df['Volume']
                    })
                    ohlc_data.dropna(inplace=True)
                    if ohlc_data.empty: continue

                    clean_close = ohlc_data['Close'] 
                    price_now = float(clean_close.iloc[-1])
                    ma5 = float(clean_close.rolling(5).mean().iloc[-1])
                    ma20 = float(clean_close.rolling(20).mean().iloc[-1])
                    bias_20 = ((price_now - ma20) / ma20) * 100
                    k, d, k_prev, d_prev = calculate_kd(clean_close, ohlc_data['High'], ohlc_data['Low'])
                    
                    # 這裡傳入原始數字代號給 FinMind 即可
                    f_buy, t_buy, yoy, yoy_str, rev_amt = get_finmind_data(stock_code)

                    processed_data.append({
                        "stock": valid_ticker, # 顯示正確的 .TW 或 .TWO
                        "price_now": price_now, "bias_20": bias_20,
                        "k": k, "d": d, "f_buy": f_buy, "t_buy": t_buy, "yoy_str": yoy_str,
                        "data_close": clean_close 
                    })

                    # PDF
                    pdf.add_page()
                    pdf.set_font("Arial", 'B', 16)
                    pdf.cell(0, 10, f"Stock Symbol: {valid_ticker}", 0, 1)
                    pdf.set_font("Arial", '', 12)
                    pdf.cell(0, 8, f"Price: {price_now:.2f} | Bias(20MA): {bias_20:.2f}%", 0, 1)
                    pdf.cell(0, 8, f"KD: K={k:.1f}, D={d:.1f}", 0, 1)
                    pdf.cell(0, 8, f"Chips(5d): Foreign {int(f_buy)}, Trust {int(t_buy)}", 0, 1)
                    pdf.cell(0, 8, f"Revenue YoY: {yoy_str.replace('No Base', 'N/A')}", 0, 1)
                    
                    mc = mpf.make_marketcolors(up='r', down='g', inherit=True)
                    s  = mpf.make_mpf_style(base_mpf_style='yahoo', marketcolors=mc)
                    subset_ohlc = ohlc_data.tail(120)
                    
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmpfile:
                        mpf.plot(subset_ohlc, type='candle', mav=(5, 20), volume=True, style=s, 
                                 title=f"\n{valid_ticker} Daily Chart (Last 6 Months)", 
                                 savefig=dict(fname=tmpfile.name, dpi=100, pad_inches=0.25))
                        tmp_filename = tmpfile.name
                    pdf.image(tmp_filename, x=10, y=60, w=190)
                    os.unlink(tmp_filename)
                    pdf.ln(120) 

                    # AI Data
                    ai_data = {
                        "symbol": valid_ticker,
                        "timestamp": datetime.now().strftime('%Y-%m-%d %H:%M'),
                        "price_data": {"current": round(price_now, 2), "bias_20_pct": round(bias_20, 2)},
                        "chips_data": {"foreign_net_buy_5d": float(f_buy), "trust_net_buy_5d": float(t_buy)},
                        "fundamentals": {"monthly_revenue_yoy_pct": round(yoy, 2) if yoy is not None else None},
                        "indicators": {"k_value": round(k, 2), "d_value": round(d, 2)}
                    }
                    all_ai_data_list.append(ai_data)

                except Exception as e:
                    st.error(f"處理 {valid_ticker} 時發生錯誤: {e}")
                    continue
            
            st.session_state['analyzed_data'] = processed_data
            st.session_state['json_txt'] = json.dumps(all_ai_data_list, indent=4, ensure_ascii=False)
            try:
                st.session_state['pdf_bytes'] = pdf.output(dest='S').encode('latin-1')
            except:
                st.session_state['pdf_bytes'] = None

    st.info("提示：輸入代號即可，系統會自動判斷上市或上櫃")

# --- 顯示邏輯 ---
if st.session_state['analyzed_data']:
    col_d1, col_d2 = st.columns(2)
    if st.session_state['json_txt']:
        with col_d1:
            st.download_button(
                label="📥 下載 AI 數據包 (.txt)",
                data=st.session_state['json_txt'],
                file_name=f"gemini_stock_data_{datetime.now().strftime('%Y%m%d')}.txt",
                mime="text/plain"
            )

    if st.session_state['pdf_bytes']:
        with col_d2:
            st.download_button(
                label="📥 下載視覺化報告 (.pdf)",
                data=st.session_state['pdf_bytes'],
                file_name=f"gemini_stock_report_{datetime.now().strftime('%Y%m%d')}.pdf",
                mime="application/pdf"
            )
    
    st.divider()

    for item in st.session_state['analyzed_data']:
        st.subheader(f"🔷 {item['stock']}")
        col1, col2, col3, col4 = st.columns(4)
        with col1: st.metric("最新股價", f"{item['
