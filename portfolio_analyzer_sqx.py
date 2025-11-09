# SCRIPT MEJORADO: PORTFOLIO ANALYZER N ESTRATEGIAS
# Soporta 2-10+ estrategias con selector dinámico
# Archivo: portfolio_analyzer_v2_multiestrategy.py

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import spearmanr, pearsonr
import warnings
warnings.filterwarnings('ignore')

st.set_page_config(page_title="Portfolio Analyzer V2 - Multi Strategy", layout="wide")
st.title("🎯 Portfolio Analyzer V2 - Multi Strategy")
st.markdown("**Análisis avanzado para N estrategias con correlación dinámica**")

# ========================================================================
# FUNCIONES AUXILIARES
# ========================================================================

def parse_sqx_csv(uploaded_file):
    """Lee CSV de SQX"""
    try:
        for sep in [';', ',', '\t']:
            try:
                df = pd.read_csv(uploaded_file, sep=sep, encoding='utf-8')
                if len(df.columns) > 3:
                    break
            except:
                uploaded_file.seek(0)
                continue
        
        df.columns = df.columns.str.strip().str.replace('"', '').str.lower()
        
        rename_map = {
            'open time': 'open_time', 'open price': 'open_price',
            'close time': 'close_time', 'close price': 'close_price',
            'profit/loss': 'pnl', 'size': 'volume',
        }
        df = df.rename(columns=rename_map)
        
        df['open_time'] = pd.to_datetime(df['open_time'], errors='coerce')
        df['close_time'] = pd.to_datetime(df['close_time'], errors='coerce')
        df['pnl'] = pd.to_numeric(df['pnl'], errors='coerce')
        df['balance'] = pd.to_numeric(df['balance'], errors='coerce')
        df = df.sort_values('close_time').reset_index(drop=True)
        
        return df[['open_time', 'close_time', 'symbol', 'type', 'volume', 'open_price', 'close_price', 'pnl', 'balance', 'comment']].copy()
    except Exception as e:
        st.error(f"Error: {e}")
        return None

def calculate_drawdown_series(pnl_series):
    """Calcula drawdown"""
    cumulative = pnl_series.cumsum()
    running_max = cumulative.cummax()
    drawdown = running_max - cumulative
    drawdown_pct = (drawdown / running_max.replace(0, np.nan)) * 100
    return drawdown, drawdown_pct.fillna(0), running_max

def calculate_strategy_metrics(df):
    """Métricas de estrategia"""
    pnl = df['pnl'].fillna(0)
    total_pnl = pnl.sum()
    num_trades = len(pnl)
    win_rate = (pnl > 0).sum() / num_trades * 100 if num_trades > 0 else 0
    gains = pnl[pnl > 0].sum()
    losses = abs(pnl[pnl < 0].sum())
    pf = gains / losses if losses > 0 else (np.inf if gains > 0 else 0)
    cumulative = pnl.cumsum()
    running_max = cumulative.cummax()
    max_dd = (running_max - cumulative).max()
    sharpe = pnl.mean() / pnl.std() * np.sqrt(252) if pnl.std() > 0 else 0
    return {
        'total_pnl': total_pnl, 'num_trades': num_trades, 'win_rate': win_rate,
        'profit_factor': pf, 'max_dd': max_dd, 'sharpe': sharpe, 'avg_trade': pnl.mean(),
    }

def calculate_all_correlations(strategies_dict):
    """Correlación NxN entre todas las estrategias"""
    strategy_names = list(strategies_dict.keys())
    n = len(strategy_names)
    
    results = {
        'pearson': np.zeros((n, n)),
        'spearman': np.zeros((n, n)),
        'sync_ratio': np.zeros((n, n)),
        'joint_dd_pct': np.zeros((n, n)),
        'dd_timing': np.zeros((n, n)),
    }
    
    # Calcular DD para todas
    dd_series = {}
    for strat_name, df in strategies_dict.items():
        pnl = df['pnl'].fillna(0).values
        _, dd_pct, _ = calculate_drawdown_series(pd.Series(pnl))
        dd_series[strat_name] = dd_pct.values
    
    # Correlaciones entre todos los pares
    for i, strat1 in enumerate(strategy_names):
        for j, strat2 in enumerate(strategy_names):
            if i == j:
                results['pearson'][i, j] = 1.0
                results['spearman'][i, j] = 1.0
                results['sync_ratio'][i, j] = 1.0
                results['joint_dd_pct'][i, j] = 100.0
                results['dd_timing'][i, j] = 1.0
            else:
                dd1 = dd_series[strat1]
                dd2 = dd_series[strat2]
                
                min_len = min(len(dd1), len(dd2))
                dd1 = dd1[:min_len]
                dd2 = dd2[:min_len]
                
                if min_len > 1:
                    try:
                        results['pearson'][i, j], _ = pearsonr(dd1, dd2)
                        results['spearman'][i, j], _ = spearmanr(dd1, dd2)
                    except:
                        results['pearson'][i, j] = 0
                        results['spearman'][i, j] = 0
                
                in_dd1 = dd1 > 0.5
                in_dd2 = dd2 > 0.5
                simultaneous = (in_dd1 & in_dd2).sum()
                any_dd = (in_dd1 | in_dd2).sum()
                sync_ratio = simultaneous / any_dd if any_dd > 0 else 0
                results['sync_ratio'][i, j] = sync_ratio
                results['joint_dd_pct'][i, j] = simultaneous / min_len * 100 if min_len > 0 else 0
                
                dd1_peaks = (dd1 > np.roll(dd1, 1)) & (dd1 > np.roll(dd1, -1))
                dd2_peaks = (dd2 > np.roll(dd2, 1)) & (dd2 > np.roll(dd2, -1))
                shared_peaks = (dd1_peaks & dd2_peaks).sum()
                all_peaks = (dd1_peaks | dd2_peaks).sum()
                results['dd_timing'][i, j] = shared_peaks / all_peaks if all_peaks > 0 else 0
    
    return results, strategy_names

def calculate_rolling_correlation_pair(df1, df2, window_size=30, method='pearson'):
    """Rolling correlation entre 2 estrategias"""
    pnl1 = df1['pnl'].fillna(0).values
    pnl2 = df2['pnl'].fillna(0).values
    
    _, dd_pct1, _ = calculate_drawdown_series(pd.Series(pnl1))
    _, dd_pct2, _ = calculate_drawdown_series(pd.Series(pnl2))
    
    min_len = min(len(dd_pct1), len(dd_pct2))
    dd_pct1 = dd_pct1[:min_len]
    dd_pct2 = dd_pct2[:min_len]
    
    correlations = []
    for i in range(window_size, min_len):
        w1 = dd_pct1[i-window_size:i]
        w2 = dd_pct2[i-window_size:i]
        try:
            if method == 'pearson':
                corr, _ = pearsonr(w1, w2)
            elif method == 'spearman':
                corr, _ = spearmanr(w1, w2)
            else:
                in_dd1 = w1 > 0.5
                in_dd2 = w2 > 0.5
                corr = (in_dd1 & in_dd2).sum() / (in_dd1 | in_dd2).sum() if (in_dd1 | in_dd2).sum() > 0 else 0
        except:
            corr = 0
        correlations.append(corr)
    
    return pd.DataFrame({'trade_number': range(len(correlations)), 'correlation': correlations})

def calculate_monthly_correlation_pair(df1, df2, method='pearson'):
    """Correlación mensual entre 2 estrategias"""
    if 'close_time' not in df1.columns or 'close_time' not in df2.columns:
        return None
    
    pnl1 = df1['pnl'].fillna(0)
    pnl2 = df2['pnl'].fillna(0)
    
    _, dd_pct1, _ = calculate_drawdown_series(pnl1)
    _, dd_pct2, _ = calculate_drawdown_series(pnl2)
    
    df1_copy = df1.copy()
    df2_copy = df2.copy()
    df1_copy['dd_pct'] = dd_pct1.values
    df2_copy['dd_pct'] = dd_pct2.values
    df1_copy['month'] = pd.to_datetime(df1_copy['close_time']).dt.to_period('M')
    df2_copy['month'] = pd.to_datetime(df2_copy['close_time']).dt.to_period('M')
    
    monthly_corrs = []
    months1 = set(df1_copy['month'].dropna())
    months2 = set(df2_copy['month'].dropna())
    
    for month in sorted(months1 & months2):
        dd1_m = df1_copy[df1_copy['month'] == month]['dd_pct'].values
        dd2_m = df2_copy[df2_copy['month'] == month]['dd_pct'].values
        min_len = min(len(dd1_m), len(dd2_m))
        
        if min_len < 5:
            continue
        
        try:
            if method == 'pearson':
                corr, _ = pearsonr(dd1_m[:min_len], dd2_m[:min_len])
            elif method == 'spearman':
                corr, _ = spearmanr(dd1_m[:min_len], dd2_m[:min_len])
            else:
                in_dd1 = dd1_m[:min_len] > 0.5
                in_dd2 = dd2_m[:min_len] > 0.5
                corr = (in_dd1 & in_dd2).sum() / (in_dd1 | in_dd2).sum() if (in_dd1 | in_dd2).sum() > 0 else 0
        except:
            corr = 0
        
        monthly_corrs.append({'month': str(month), 'correlation': corr, 'trades': min_len})
    
    return pd.DataFrame(monthly_corrs)

def recommend_portfolio_auto(metrics_dict, optimization='min_dd'):
    """Recomendación automática de portafolio"""
    metrics_list = [(name, metrics) for name, metrics in metrics_dict.items()]
    
    if optimization == 'min_dd':
        ranked = sorted(metrics_list, key=lambda x: x[1]['max_dd'])
        top_n = min(3, len(ranked))
        selected = [name for name, _ in ranked[:top_n]]
        return selected, f"Top {top_n} por menor DD"
    
    elif optimization == 'max_pf':
        ranked = sorted(metrics_list, key=lambda x: x[1]['profit_factor'], reverse=True)
        top_n = min(3, len(ranked))
        selected = [name for name, _ in ranked[:top_n]]
        return selected, f"Top {top_n} por mayor Profit Factor"
    
    elif optimization == 'max_sharpe':
        ranked = sorted(metrics_list, key=lambda x: x[1]['sharpe'], reverse=True)
        top_n = min(3, len(ranked))
        selected = [name for name, _ in ranked[:top_n]]
        return selected, f"Top {top_n} por mayor Sharpe"
    
    return list(metrics_dict.keys()), "Todas las estrategias"

def calculate_portfolio_metrics(strategies_dict, selected_strategies, weights=None):
    """Calcula métricas de portafolio combinado"""
    if weights is None:
        weights = {name: 1/len(selected_strategies) for name in selected_strategies}
    
    pnl_combined = []
    for strat_name in selected_strategies:
        if strat_name in strategies_dict:
            df = strategies_dict[strat_name]
            pnl = df['pnl'].fillna(0).values * weights[strat_name]
            pnl_combined.extend(pnl)
    
    pnl_combined = pd.Series(pnl_combined)
    cumulative = pnl_combined.cumsum()
    running_max = cumulative.cummax()
    max_dd = (running_max - cumulative).max()
    
    return {
        'total_pnl': pnl_combined.sum(),
        'max_dd': max_dd,
        'win_rate': (pnl_combined > 0).sum() / len(pnl_combined) * 100 if len(pnl_combined) > 0 else 0,
        'sharpe': pnl_combined.mean() / pnl_combined.std() * np.sqrt(252) if pnl_combined.std() > 0 else 0,
        'num_trades': len(pnl_combined),
    }

# ========================================================================
# STREAMLIT UI
# ========================================================================

st.sidebar.header("📁 Cargar Estrategias")
uploaded_files = st.sidebar.file_uploader("📤 CSV de SQX", type=['csv'], accept_multiple_files=True)

if not uploaded_files:
    st.info("⏳ Carga 2-10 archivos CSV")
    st.stop()

strategies_dict = {}
for uploaded_file in uploaded_files:
    df = parse_sqx_csv(uploaded_file)
    if df is not None and not df.empty:
        strat_name = uploaded_file.name.replace('.csv', '').replace('.CSV', '')
        strategies_dict[strat_name] = df

if len(strategies_dict) < 2:
    st.error("❌ Necesitas al menos 2 estrategias")
    st.stop()

st.success(f"✅ {len(strategies_dict)} estrategias cargadas")

# ========================================================================
# TABS
# ========================================================================

tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
    "📊 Resumen", 
    "🔗 Correlación", 
    "📈 Dinámica",
    "🎯 Selector Portfolio",
    "🎨 Multi-Portfolio",
    "📈 Análisis",
    "💡 Insights"
])

# ========================================================================
# TAB 1: RESUMEN (TODAS LAS ESTRATEGIAS)
# ========================================================================

with tab1:
    st.header("📊 Resumen de Todas las Estrategias")
    
    summary_data = []
    for strat_name, df in strategies_dict.items():
        metrics = calculate_strategy_metrics(df)
        summary_data.append({
            'Estrategia': strat_name,
            'PnL': metrics['total_pnl'],
            'Trades': metrics['num_trades'],
            'Win %': metrics['win_rate'],
            'PF': metrics['profit_factor'],
            'DD': metrics['max_dd'],
            'Sharpe': metrics['sharpe'],
        })
    
    summary_df = pd.DataFrame(summary_data).sort_values('PnL', ascending=False)
    
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("### Por PnL (Mayor a Menor)")
        st.dataframe(
            summary_df.sort_values('PnL', ascending=False).style.format({
                'PnL': '{:.2f}', 'Win %': '{:.1f}%', 'PF': '{:.2f}', 'DD': '{:.2f}', 'Sharpe': '{:.2f}'
            }),
            use_container_width=True, hide_index=True
        )
    
    with col2:
        st.markdown("### Por DD Menor (Mejor Riesgo)")
        st.dataframe(
            summary_df.sort_values('DD').style.format({
                'PnL': '{:.2f}', 'Win %': '{:.1f}%', 'PF': '{:.2f}', 'DD': '{:.2f}', 'Sharpe': '{:.2f}'
            }),
            use_container_width=True, hide_index=True
        )
    
    # Gráficos comparativos
    col1, col2, col3 = st.columns(3)
    
    with col1:
        fig, ax = plt.subplots(figsize=(8, 5))
        summary_df.plot(x='Estrategia', y='PnL', kind='barh', ax=ax, color='green')
        plt.title('PnL por Estrategia')
        st.pyplot(fig, use_container_width=True)
    
    with col2:
        fig, ax = plt.subplots(figsize=(8, 5))
        summary_df.plot(x='Estrategia', y='DD', kind='barh', ax=ax, color='red')
        plt.title('Max DD por Estrategia')
        st.pyplot(fig, use_container_width=True)
    
    with col3:
        fig, ax = plt.subplots(figsize=(8, 5))
        summary_df.plot(x='Estrategia', y='Sharpe', kind='barh', ax=ax, color='blue')
        plt.title('Sharpe Ratio por Estrategia')
        st.pyplot(fig, use_container_width=True)

# ========================================================================
# TAB 2: CORRELACIÓN (MATRIZ NxN)
# ========================================================================

with tab2:
    st.header("🔗 Matriz de Correlación (NxN)")
    
    corr_method = st.selectbox("Método:", ['pearson', 'spearman', 'sync_ratio', 'joint_dd_pct', 'dd_timing'])
    
    corr_results, strat_names = calculate_all_correlations(strategies_dict)
    
    corr_map = {'pearson': corr_results['pearson'], 'spearman': corr_results['spearman'],
                'sync_ratio': corr_results['sync_ratio'], 'joint_dd_pct': corr_results['joint_dd_pct'],
                'dd_timing': corr_results['dd_timing']}
    
    corr_matrix = corr_map[corr_method]
    
    fig, ax = plt.subplots(figsize=(12, 10))
    sns.heatmap(corr_matrix, xticklabels=strat_names, yticklabels=strat_names,
                annot=True, fmt='.2f', cmap='RdYlGn_r', center=0.5, ax=ax)
    plt.title(f'Correlación {corr_method.upper()} - Todas las Estrategias')
    st.pyplot(fig, use_container_width=True)
    
    # Tabla de correlaciones
    st.markdown("### Pares Ordenados por Pearson")
    pairs_data = []
    n = len(strat_names)
    for i in range(n):
        for j in range(i+1, n):
            pairs_data.append({
                'Est 1': strat_names[i],
                'Est 2': strat_names[j],
                'Pearson': corr_results['pearson'][i, j],
                'Spearman': corr_results['spearman'][i, j],
                'Sync': corr_results['sync_ratio'][i, j],
            })
    
    pairs_df = pd.DataFrame(pairs_data).sort_values('Pearson')
    st.dataframe(pairs_df.style.format({col: '{:.3f}' for col in ['Pearson', 'Spearman', 'Sync']}),
                 use_container_width=True, hide_index=True)

# ========================================================================
# TAB 3: CORRELACIÓN DINÁMICA
# ========================================================================

with tab3:
    st.header("📈 Correlación Dinámica (Elige 2 Estrategias)")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        strat1 = st.selectbox("Estrategia 1:", list(strategies_dict.keys()), key='dyn1')
    with col2:
        strat2_options = [s for s in strategies_dict.keys() if s != strat1]
        strat2 = st.selectbox("Estrategia 2:", strat2_options, key='dyn2')
    with col3:
        analysis_type = st.selectbox("Tipo:", ['Rolling', 'Mensual'])
    
    if analysis_type == 'Rolling':
        window = st.slider("Ventana (trades):", 10, 100, 30)
        result_df = calculate_rolling_correlation_pair(strategies_dict[strat1], strategies_dict[strat2], window)
        x_label = 'Trade #'
    else:
        result_df = calculate_monthly_correlation_pair(strategies_dict[strat1], strategies_dict[strat2])
        x_label = 'Mes'
    
    if result_df is not None and not result_df.empty:
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Promedio", f"{result_df['correlation'].mean():.3f}")
        with col2:
            st.metric("Máximo", f"{result_df['correlation'].max():.3f}")
        with col3:
            st.metric("Mínimo", f"{result_df['correlation'].min():.3f}")
        with col4:
            st.metric("Desv", f"{result_df['correlation'].std():.3f}")
        
        fig, ax = plt.subplots(figsize=(14, 6))
        ax.plot(result_df['correlation'], linewidth=2, label='Correlación')
        ax.axhline(result_df['correlation'].mean(), color='red', linestyle='--', label='Promedio')
        ax.axhspan(0.7, 1, alpha=0.2, color='red', label='Alta (>0.7)')
        ax.axhspan(-1, 0.3, alpha=0.2, color='green', label='Baja (<0.3)')
        ax.set_title(f'Correlación Dinámica: {strat1} vs {strat2}')
        ax.legend()
        ax.grid(True, alpha=0.3)
        st.pyplot(fig, use_container_width=True)
        
        st.dataframe(result_df.style.format({'correlation': '{:.3f}'}), use_container_width=True)

# ========================================================================
# TAB 4: SELECTOR PORTFOLIO
# ========================================================================

with tab4:
    st.header("🎯 Selector de Portfolio")
    
    selection_mode = st.radio("Modo de Selección:", 
        ['Auto (Recomendación)', 'Manual (Elige N)', 'Filtrado (Criterios)'])
    
    if selection_mode == 'Auto (Recomendación)':
        opt_type = st.selectbox("Optimizar por:", ['Menor DD', 'Mayor Profit Factor', 'Mayor Sharpe'])
        opt_map = {'Menor DD': 'min_dd', 'Mayor Profit Factor': 'max_pf', 'Mayor Sharpe': 'max_sharpe'}
        
        metrics_dict = {name: calculate_strategy_metrics(df) for name, df in strategies_dict.items()}
        selected, reason = recommend_portfolio_auto(metrics_dict, opt_map[opt_type])
        
        st.success(f"✅ Recomendación: {reason}")
        st.info(f"Estrategias seleccionadas: {', '.join(selected)}")
        
        st.session_state.selected_strategies = selected
    
    elif selection_mode == 'Manual (Elige N)':
        selected = st.multiselect("Elige estrategias:", list(strategies_dict.keys()), default=list(strategies_dict.keys())[:2])
        st.session_state.selected_strategies = selected
        st.success(f"Seleccionadas {len(selected)} estrategias")
    
    else:  # Filtrado
        st.markdown("### Establece Criterios")
        max_dd = st.slider("DD máximo permitido:", 0.0, 10000.0, 5000.0)
        min_pf = st.slider("Profit Factor mínimo:", 0.5, 5.0, 1.5)
        max_corr = st.slider("Correlación máxima:", 0.0, 1.0, 0.7)
        
        metrics_dict = {name: calculate_strategy_metrics(df) for name, df in strategies_dict.items()}
        
        candidates = [name for name, m in metrics_dict.items() if m['max_dd'] <= max_dd and m['profit_factor'] >= min_pf]
        
        st.session_state.selected_strategies = candidates
        st.success(f"Candidatos: {', '.join(candidates) if candidates else 'Ninguno'}")

# ========================================================================
# TAB 5: MULTI-PORTFOLIO (COMPARACIÓN)
# ========================================================================

with tab5:
    st.header("🎨 Comparar Múltiples Portfolios")
    
    portfolios_to_compare = []
    
    # Portfolio 1: Recomendación por DD
    metrics_dict = {name: calculate_strategy_metrics(df) for name, df in strategies_dict.items()}
    selected_dd, _ = recommend_portfolio_auto(metrics_dict, 'min_dd')
    portfolio_dd = calculate_portfolio_metrics(strategies_dict, selected_dd)
    portfolios_to_compare.append(('Min DD (Top 3)', selected_dd, portfolio_dd))
    
    # Portfolio 2: Recomendación por PF
    selected_pf, _ = recommend_portfolio_auto(metrics_dict, 'max_pf')
    portfolio_pf = calculate_portfolio_metrics(strategies_dict, selected_pf)
    portfolios_to_compare.append(('Max PF (Top 3)', selected_pf, portfolio_pf))
    
    # Portfolio 3: Recomendación por Sharpe
    selected_sharpe, _ = recommend_portfolio_auto(metrics_dict, 'max_sharpe')
    portfolio_sharpe = calculate_portfolio_metrics(strategies_dict, selected_sharpe)
    portfolios_to_compare.append(('Max Sharpe (Top 3)', selected_sharpe, portfolio_sharpe))
    
    # Portfolio 4: Todas
    selected_all = list(strategies_dict.keys())
    portfolio_all = calculate_portfolio_metrics(strategies_dict, selected_all)
    portfolios_to_compare.append(('Todas', selected_all, portfolio_all))
    
    # Tabla comparativa
    comparison_data = []
    for port_name, strategies, metrics in portfolios_to_compare:
        comparison_data.append({
            'Portfolio': port_name,
            'Estrategias': ', '.join(strategies),
            'PnL': metrics['total_pnl'],
            'DD': metrics['max_dd'],
            'Win %': metrics['win_rate'],
            'Sharpe': metrics['sharpe'],
        })
    
    comp_df = pd.DataFrame(comparison_data)
    st.dataframe(
        comp_df.style.format({'PnL': '{:.2f}', 'DD': '{:.2f}', 'Win %': '{:.1f}%', 'Sharpe': '{:.2f}'}),
        use_container_width=True, hide_index=True
    )
    
    # Gráficos comparativos
    col1, col2 = st.columns(2)
    with col1:
        fig, ax = plt.subplots(figsize=(10, 6))
        comp_df.plot(x='Portfolio', y=['PnL', 'DD'], kind='bar', ax=ax)
        plt.title('PnL vs DD por Portfolio')
        plt.xticks(rotation=45)
        st.pyplot(fig, use_container_width=True)
    
    with col2:
        fig, ax = plt.subplots(figsize=(10, 6))
        comp_df.plot(x='Portfolio', y='Sharpe', kind='bar', ax=ax, color='purple')
        plt.title('Sharpe Ratio por Portfolio')
        plt.xticks(rotation=45)
        st.pyplot(fig, use_container_width=True)

# ========================================================================
# TAB 6: ANÁLISIS DETALLADO
# ========================================================================

with tab6:
    st.header("📈 Análisis Detallado")
    
    selected_strat = st.selectbox("Elige estrategia:", list(strategies_dict.keys()))
    df_selected = strategies_dict[selected_strat]
    pnl = df_selected['pnl'].fillna(0)
    equity = pnl.cumsum()
    
    fig, ax = plt.subplots(figsize=(14, 6))
    ax.plot(equity.values, linewidth=2)
    ax.fill_between(range(len(equity)), equity.values, alpha=0.3)
    ax.set_title(f'Equity Curve - {selected_strat}')
    ax.grid(True, alpha=0.3)
    st.pyplot(fig, use_container_width=True)

# ========================================================================
# TAB 7: INSIGHTS
# ========================================================================

with tab7:
    st.header("💡 Insights Finales")
    
    st.markdown("### Recomendación del Sistema")
    st.info("""
    ✅ **Mejor Portfolio Recomendado:**
    - Estrategias: [Top 3 por menor DD]
    - Razón: Minimizar riesgo agregado
    - DD Esperado: [X% menor que individual]
    """)
