import streamlit as st
import pandas as pd
import numpy as np
from io import BytesIO
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import spearmanr, pearsonr
import warnings
warnings.filterwarnings('ignore')

st.set_page_config(page_title="Portfolio Analyzer SQX - Drawdown Correlation", layout="wide")
st.title("🎯 Portfolio Analyzer SQX - Correlación por Drawdown")
st.markdown("**Análisis avanzado de correlación de estrategias basado en sincronización de drawdowns**")

# ========================================================================
# FUNCIONES DE CÁLCULO
# ========================================================================

def parse_sqx_csv(uploaded_file):
    """Lee CSV exportado directamente de SQX"""
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
            'open time': 'open_time',
            'open price': 'open_price',
            'close time': 'close_time',
            'close price': 'close_price',
            'profit/loss': 'pnl',
            'size': 'volume',
        }
        df = df.rename(columns=rename_map)
        
        df['open_time'] = pd.to_datetime(df['open_time'], errors='coerce')
        df['close_time'] = pd.to_datetime(df['close_time'], errors='coerce')
        df['pnl'] = pd.to_numeric(df['pnl'], errors='coerce')
        df['balance'] = pd.to_numeric(df['balance'], errors='coerce')
        
        df = df.sort_values('close_time').reset_index(drop=True)
        
        return df[['open_time', 'close_time', 'symbol', 'type', 'volume', 
                  'open_price', 'close_price', 'pnl', 'balance', 'comment']].copy()
    
    except Exception as e:
        st.error(f"Error parsing CSV: {e}")
        return None

def calculate_drawdown_series(pnl_series):
    """Calcula la serie de drawdown"""
    cumulative = pnl_series.cumsum()
    running_max = cumulative.cummax()
    drawdown = running_max - cumulative
    drawdown_pct = (drawdown / running_max.replace(0, np.nan)) * 100
    drawdown_pct = drawdown_pct.fillna(0)
    return drawdown, drawdown_pct, running_max

def calculate_strategy_metrics(df):
    """Calcula métricas principales"""
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
    
    if pnl.std() > 0:
        sharpe = pnl.mean() / pnl.std() * np.sqrt(252)
    else:
        sharpe = 0
    
    return {
        'total_pnl': total_pnl,
        'num_trades': num_trades,
        'win_rate': win_rate,
        'profit_factor': pf,
        'max_dd': max_dd,
        'sharpe': sharpe,
        'avg_trade': pnl.mean(),
    }

def calculate_dd_correlation_methods(strategies_dict):
    """Calcula correlaciones de drawdown (VERSIÓN CORREGIDA)"""
    
    strategy_names = list(strategies_dict.keys())
    n = len(strategy_names)
    
    results = {
        'pearson': np.zeros((n, n)),
        'spearman': np.zeros((n, n)),
        'sync_ratio': np.zeros((n, n)),
        'joint_dd_pct': np.zeros((n, n)),
        'dd_timing': np.zeros((n, n)),
        'drawdown_series': {},
        'drawdown_pct': {},
    }
    
    for strat_name, df in strategies_dict.items():
        pnl = df['pnl'].fillna(0).values
        dd, dd_pct, _ = calculate_drawdown_series(pd.Series(pnl))
        results['drawdown_series'][strat_name] = dd.values
        results['drawdown_pct'][strat_name] = dd_pct.values
    
    for i, strat1 in enumerate(strategy_names):
        for j, strat2 in enumerate(strategy_names):
            if i == j:
                results['pearson'][i, j] = 1.0
                results['spearman'][i, j] = 1.0
                results['sync_ratio'][i, j] = 1.0
                results['joint_dd_pct'][i, j] = 100.0
                results['dd_timing'][i, j] = 1.0
            else:
                dd1 = results['drawdown_pct'][strat1]
                dd2 = results['drawdown_pct'][strat2]
                
                # ✅ CORRECCIÓN: Alinear series
                min_len = min(len(dd1), len(dd2))
                dd1_aligned = dd1[:min_len]
                dd2_aligned = dd2[:min_len]
                
                if min_len > 1:
                    try:
                        pearson_val, _ = pearsonr(dd1_aligned, dd2_aligned)
                        results['pearson'][i, j] = pearson_val
                    except:
                        results['pearson'][i, j] = 0
                    
                    try:
                        spearman_val, _ = spearmanr(dd1_aligned, dd2_aligned)
                        results['spearman'][i, j] = spearman_val
                    except:
                        results['spearman'][i, j] = 0
                
                in_dd1 = dd1_aligned > 0.5
                in_dd2 = dd2_aligned > 0.5
                simultaneous_dd = (in_dd1 & in_dd2).sum()
                any_dd = (in_dd1 | in_dd2).sum()
                sync_ratio = simultaneous_dd / any_dd if any_dd > 0 else 0
                results['sync_ratio'][i, j] = sync_ratio
                
                joint_dd = (in_dd1 & in_dd2).sum() / min_len * 100 if min_len > 0 else 0
                results['joint_dd_pct'][i, j] = joint_dd
                
                dd1_peaks = (dd1_aligned > np.roll(dd1_aligned, 1)) & (dd1_aligned > np.roll(dd1_aligned, -1))
                dd2_peaks = (dd2_aligned > np.roll(dd2_aligned, 1)) & (dd2_aligned > np.roll(dd2_aligned, -1))
                shared_peaks = (dd1_peaks & dd2_peaks).sum()
                all_peaks = (dd1_peaks | dd2_peaks).sum()
                timing_corr = shared_peaks / all_peaks if all_peaks > 0 else 0
                results['dd_timing'][i, j] = timing_corr
    
    return results, strategy_names

def calculate_rolling_dd_correlation(strategies_dict, window_size=30, method='pearson'):
    """Correlación dinámica en ventanas móviles"""
    
    strategy_names = list(strategies_dict.keys())
    
    if len(strategy_names) < 2:
        return None, None
    
    strat1_name = strategy_names[0]
    strat2_name = strategy_names[1]
    
    df1 = strategies_dict[strat1_name]
    df2 = strategies_dict[strat2_name]
    
    pnl1 = df1['pnl'].fillna(0).values
    pnl2 = df2['pnl'].fillna(0).values
    
    _, dd_pct1, _ = calculate_drawdown_series(pd.Series(pnl1))
    _, dd_pct2, _ = calculate_drawdown_series(pd.Series(pnl2))
    
    min_len = min(len(dd_pct1), len(dd_pct2))
    dd_pct1 = dd_pct1[:min_len].values
    dd_pct2 = dd_pct2[:min_len].values
    
    correlations = []
    periods = []
    dates = []
    
    for i in range(window_size, min_len):
        window_dd1 = dd_pct1[i-window_size:i]
        window_dd2 = dd_pct2[i-window_size:i]
        
        try:
            if method == 'pearson':
                corr, _ = pearsonr(window_dd1, window_dd2)
            elif method == 'spearman':
                corr, _ = spearmanr(window_dd1, window_dd2)
            elif method == 'sync_ratio':
                in_dd1 = window_dd1 > 0.5
                in_dd2 = window_dd2 > 0.5
                simultaneous = (in_dd1 & in_dd2).sum()
                any_dd = (in_dd1 | in_dd2).sum()
                corr = simultaneous / any_dd if any_dd > 0 else 0
            else:
                corr = 0
        except:
            corr = 0
        
        correlations.append(corr)
        periods.append(i)
        
        if 'close_time' in df1.columns and i < len(df1):
            dates.append(df1['close_time'].iloc[i])
        else:
            dates.append(None)
    
    result_df = pd.DataFrame({
        'trade_number': periods,
        'correlation': correlations,
        'date': dates
    })
    
    metadata = {
        'strat1': strat1_name,
        'strat2': strat2_name,
        'window_size': window_size,
        'method': method
    }
    
    return result_df, metadata

def calculate_monthly_dd_correlation(strategies_dict, method='pearson'):
    """Correlación agrupada por mes"""
    
    strategy_names = list(strategies_dict.keys())
    
    if len(strategy_names) < 2:
        return None, None
    
    strat1_name = strategy_names[0]
    strat2_name = strategy_names[1]
    
    df1 = strategies_dict[strat1_name].copy()
    df2 = strategies_dict[strat2_name].copy()
    
    if 'close_time' not in df1.columns or 'close_time' not in df2.columns:
        return None, None
    
    pnl1 = df1['pnl'].fillna(0)
    pnl2 = df2['pnl'].fillna(0)
    
    _, dd_pct1, _ = calculate_drawdown_series(pnl1)
    _, dd_pct2, _ = calculate_drawdown_series(pnl2)
    
    df1['dd_pct'] = dd_pct1.values
    df2['dd_pct'] = dd_pct2.values
    
    df1['month'] = pd.to_datetime(df1['close_time']).dt.to_period('M')
    df2['month'] = pd.to_datetime(df2['close_time']).dt.to_period('M')
    
    monthly_corrs = []
    
    months1 = set(df1['month'].dropna())
    months2 = set(df2['month'].dropna())
    common_months = sorted(months1 & months2)
    
    for month in common_months:
        dd1_month = df1[df1['month'] == month]['dd_pct'].values
        dd2_month = df2[df2['month'] == month]['dd_pct'].values
        
        min_len = min(len(dd1_month), len(dd2_month))
        if min_len < 5:
            continue
        
        dd1_month = dd1_month[:min_len]
        dd2_month = dd2_month[:min_len]
        
        try:
            if method == 'pearson':
                corr, _ = pearsonr(dd1_month, dd2_month)
            elif method == 'spearman':
                corr, _ = spearmanr(dd1_month, dd2_month)
            elif method == 'sync_ratio':
                in_dd1 = dd1_month > 0.5
                in_dd2 = dd2_month > 0.5
                simultaneous = (in_dd1 & in_dd2).sum()
                any_dd = (in_dd1 | in_dd2).sum()
                corr = simultaneous / any_dd if any_dd > 0 else 0
            else:
                corr = 0
        except:
            corr = 0
        
        monthly_corrs.append({
            'month': str(month),
            'correlation': corr,
            'n_trades_1': len(dd1_month),
            'n_trades_2': len(dd2_month)
        })
    
    result_df = pd.DataFrame(monthly_corrs)
    
    metadata = {
        'strat1': strat1_name,
        'strat2': strat2_name,
        'method': method
    }
    
    return result_df, metadata

def portfolio_allocation_minimum_dd(strategies_dict, method='uniform'):
    """Sugiere asignación de portafolio"""
    
    metrics = {}
    for name, df in strategies_dict.items():
        m = calculate_strategy_metrics(df)
        metrics[name] = m
    
    if method == 'uniform':
        n = len(metrics)
        return {name: 1/n for name in metrics.keys()}
    
    elif method == 'inverse_dd':
        dds = {name: m['max_dd'] for name, m in metrics.items()}
        min_dd = min(dds.values())
        inverse = {name: 1/(dd - min_dd + 0.0001) for name, dd in dds.items()}
        total = sum(inverse.values())
        return {name: w/total for name, w in inverse.items()}
    
    elif method == 'sharpe':
        sharpes = {name: max(0, m['sharpe']) for name, m in metrics.items()}
        total = sum(sharpes.values())
        if total == 0:
            return {name: 1/len(metrics) for name in metrics.keys()}
        return {name: w/total for name, w in sharpes.items()}
    
    elif method == 'sortino':
        allocations = {}
        for name, m in metrics.items():
            df = strategies_dict[name]
            pnl = df['pnl'].fillna(0)
            downside = pnl[pnl < 0].std()
            sortino = m['avg_trade'] / downside if downside > 0 else 0
            allocations[name] = max(0, sortino)
        
        total = sum(allocations.values())
        if total == 0:
            return {name: 1/len(metrics) for name in metrics.keys()}
        return {name: w/total for name, w in allocations.items()}

def portfolio_combined_metrics(strategies_dict, weights):
    """Calcula métricas del portafolio combinado"""
    
    total_pnl = 0
    pnl_combined = []
    
    for strat_name, weight in weights.items():
        df = strategies_dict[strat_name]
        pnl = df['pnl'].fillna(0).values
        pnl_weighted = pnl * weight
        total_pnl += pnl_weighted.sum()
        pnl_combined.extend(pnl_weighted)
    
    pnl_combined = pd.Series(pnl_combined)
    
    cumulative = pnl_combined.cumsum()
    running_max = cumulative.cummax()
    max_dd = (running_max - cumulative).max()
    
    num_trades = len(pnl_combined)
    win_rate = (pnl_combined > 0).sum() / num_trades * 100 if num_trades > 0 else 0
    
    return {
        'total_pnl': total_pnl,
        'max_dd': max_dd,
        'num_trades': num_trades,
        'win_rate': win_rate,
        'sharpe': pnl_combined.mean() / pnl_combined.std() * np.sqrt(252) if pnl_combined.std() > 0 else 0,
    }

# ========================================================================
# UI STREAMLIT
# ========================================================================

st.sidebar.header("📁 Cargar Estrategias")
uploaded_files = st.sidebar.file_uploader(
    "📤 Selecciona archivos CSV (SQX export)",
    type=['csv'],
    accept_multiple_files=True
)

if not uploaded_files:
    st.info("⏳ Carga al menos 2 archivos CSV de estrategias para comenzar")
    st.stop()

strategies_dict = {}
for uploaded_file in uploaded_files:
    df = parse_sqx_csv(uploaded_file)
    if df is not None and not df.empty:
        strat_name = uploaded_file.name.replace('.csv', '').replace('.CSV', '')
        strategies_dict[strat_name] = df

if len(strategies_dict) < 2:
    st.error("❌ Se necesitan al menos 2 estrategias válidas")
    st.stop()

st.success(f"✅ {len(strategies_dict)} estrategias cargadas correctamente")

# ========================================================================
# TABS
# ========================================================================

tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "📊 Resumen", 
    "🔗 Correlación DD", 
    "📈 Correlación Dinámica",
    "🎯 Portfolio Optimizer",
    "📊 Análisis Detallado",
    "💡 Insights"
])

# ========================================================================
# TAB 1: RESUMEN
# ========================================================================

with tab1:
    st.header("Resumen de Estrategias Individuales")
    
    summary_data = []
    for strat_name, df in strategies_dict.items():
        metrics = calculate_strategy_metrics(df)
        summary_data.append({
            'Estrategia': strat_name,
            'PnL Total': metrics['total_pnl'],
            '# Trades': metrics['num_trades'],
            'Win Rate %': metrics['win_rate'],
            'Profit Factor': metrics['profit_factor'],
            'Max DD': metrics['max_dd'],
            'Sharpe': metrics['sharpe'],
        })
    
    summary_df = pd.DataFrame(summary_data).sort_values('PnL Total', ascending=False)
    
    st.dataframe(
        summary_df.style.format({
            'PnL Total': '{:.2f}',
            'Win Rate %': '{:.2f}%',
            'Profit Factor': '{:.2f}',
            'Max DD': '{:.2f}',
            'Sharpe': '{:.2f}',
        }),
        use_container_width=True,
        height=400
    )
    
    col1, col2 = st.columns(2)
    
    with col1:
        fig, ax = plt.subplots(figsize=(10, 6))
        summary_df.plot(x='Estrategia', y=['PnL Total', 'Max DD'], kind='bar', ax=ax)
        plt.title('PnL vs Max Drawdown')
        plt.xticks(rotation=45, ha='right')
        st.pyplot(fig, use_container_width=True)
    
    with col2:
        fig, ax = plt.subplots(figsize=(10, 6))
        summary_df.plot(x='Estrategia', y='Sharpe', kind='bar', ax=ax, color='green')
        plt.title('Sharpe Ratio por Estrategia')
        plt.xticks(rotation=45, ha='right')
        st.pyplot(fig, use_container_width=True)

# ========================================================================
# TAB 2: CORRELACIÓN DD
# ========================================================================

with tab2:
    st.header("🔗 Análisis de Correlación por Drawdown")
    
    st.markdown("""
    ### Métodos de Correlación:
    - **Pearson**: Correlación lineal de series de DD
    - **Spearman**: Correlación no-paramétrica
    - **Sync Ratio**: % tiempo en DD sincronizadas
    - **Joint DD %**: % tiempo en drawdown conjunto
    - **DD Timing**: Correlación de picos de DD
    """)
    
    corr_method = st.selectbox(
        "Selecciona método de correlación:",
        ['pearson', 'spearman', 'sync_ratio', 'joint_dd_pct', 'dd_timing'],
    )
    
    corr_results, strat_names = calculate_dd_correlation_methods(strategies_dict)
    
    if corr_method == 'pearson':
        corr_matrix = corr_results['pearson']
        title = "Correlación Pearson - DD Series"
    elif corr_method == 'spearman':
        corr_matrix = corr_results['spearman']
        title = "Correlación Spearman - DD Series"
    elif corr_method == 'sync_ratio':
        corr_matrix = corr_results['sync_ratio']
        title = "Ratio de Sincronización - DD Compartidos"
    elif corr_method == 'joint_dd_pct':
        corr_matrix = corr_results['joint_dd_pct']
        title = "% Tiempo en DD Conjunto"
    else:
        corr_matrix = corr_results['dd_timing']
        title = "Correlación de Timing de Picos DD"
    
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(
        corr_matrix,
        xticklabels=strat_names,
        yticklabels=strat_names,
        annot=True,
        fmt='.2f',
        cmap='RdYlGn_r',
        center=0.5,
        ax=ax,
        cbar_kws={'label': 'Correlación'}
    )
    plt.title(title, fontsize=14, fontweight='bold')
    plt.xticks(rotation=45, ha='right')
    st.pyplot(fig, use_container_width=True)
    
    st.markdown("### Matriz de Correlación")
    corr_df = pd.DataFrame(corr_matrix, index=strat_names, columns=strat_names)
    st.dataframe(
        corr_df.style.format('{:.3f}').background_gradient(cmap='RdYlGn_r'),
        use_container_width=True
    )
    
    st.markdown("### Análisis de Pares")
    
    pairs_data = []
    n = len(strat_names)
    for i in range(n):
        for j in range(i+1, n):
            pairs_data.append({
                'Estrategia 1': strat_names[i],
                'Estrategia 2': strat_names[j],
                'Pearson': corr_results['pearson'][i, j],
                'Spearman': corr_results['spearman'][i, j],
                'Sync Ratio': corr_results['sync_ratio'][i, j],
                'Joint DD %': corr_results['joint_dd_pct'][i, j],
                'DD Timing': corr_results['dd_timing'][i, j],
            })
    
    pairs_df = pd.DataFrame(pairs_data).sort_values('Pearson')
    st.dataframe(
        pairs_df.style.format({col: '{:.3f}' for col in pairs_df.columns if col not in ['Estrategia 1', 'Estrategia 2']}),
        use_container_width=True,
        height=400
    )

# ========================================================================
# TAB 3: CORRELACIÓN DINÁMICA
# ========================================================================

with tab3:
    st.header("📈 Correlación Dinámica de Drawdowns")
    
    st.markdown("""
    **Objetivo:** Analizar cómo cambia la correlación a lo largo del tiempo.
    - ✅ Detectar cambios de régimen de mercado
    - ✅ Identificar períodos de alta/baja correlación
    - ✅ Optimizar portfolio según comportamiento histórico
    """)
    
    if len(strategies_dict) < 2:
        st.warning("Se necesitan al menos 2 estrategias")
        st.stop()
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        analysis_type = st.selectbox(
            "Tipo de análisis:",
            ['Rolling (ventana móvil)', 'Mensual'],
        )
    
    with col2:
        if analysis_type == 'Rolling (ventana móvil)':
            window_size = st.slider(
                "Tamaño ventana (trades):",
                min_value=10,
                max_value=200,
                value=30,
                step=10,
            )
        else:
            window_size = 30
    
    with col3:
        corr_method_dynamic = st.selectbox(
            "Método:",
            ['pearson', 'spearman', 'sync_ratio'],
        )
    
    temp_dict = {list(strategies_dict.keys())[0]: strategies_dict[list(strategies_dict.keys())[0]],
                 list(strategies_dict.keys())[1]: strategies_dict[list(strategies_dict.keys())[1]]}
    
    if analysis_type == 'Rolling (ventana móvil)':
        result_df, metadata = calculate_rolling_dd_correlation(
            temp_dict, 
            window_size=window_size, 
            method=corr_method_dynamic
        )
    else:
        result_df, metadata = calculate_monthly_dd_correlation(
            temp_dict,
            method=corr_method_dynamic
        )
    
    if result_df is None or result_df.empty:
        st.warning("No hay suficientes datos")
        st.stop()
    
    st.markdown("### Estadísticas de Correlación Dinámica")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Correlación Promedio", f"{result_df['correlation'].mean():.3f}")
    with col2:
        st.metric("Correlación Máxima", f"{result_df['correlation'].max():.3f}")
    with col3:
        st.metric("Correlación Mínima", f"{result_df['correlation'].min():.3f}")
    with col4:
        st.metric("Desv. Estándar", f"{result_df['correlation'].std():.3f}")
    
    st.markdown("### Serie Temporal de Correlación")
    
    fig, ax = plt.subplots(figsize=(14, 6))
    
    if analysis_type == 'Rolling (ventana móvil)':
        x_values = result_df['trade_number']
        x_label = 'Trade Number'
    else:
        x_values = range(len(result_df))
        x_label = 'Período'
    
    ax.plot(x_values, result_df['correlation'], linewidth=2, color='steelblue', label='Correlación')
    ax.axhline(y=result_df['correlation'].mean(), color='red', linestyle='--', linewidth=2, label='Promedio')
    ax.axhline(y=0, color='gray', linestyle='-', linewidth=0.5, alpha=0.5)
    
    ax.axhspan(0.7, 1.0, alpha=0.2, color='red', label='Alta correlación (>0.7)')
    ax.axhspan(-1.0, 0.3, alpha=0.2, color='green', label='Baja correlación (<0.3)')
    
    ax.set_xlabel(x_label)
    ax.set_ylabel('Correlación')
    ax.set_title(f'Correlación Dinámica: {metadata["strat1"]} vs {metadata["strat2"]}')
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)
    
    st.pyplot(fig, use_container_width=True)
    
    st.markdown("### Detalle por Período")
    
    display_df = result_df.copy()
    if analysis_type == 'Mensual':
        display_df = display_df.sort_values('month', ascending=False)
    else:
        display_df = display_df.tail(50)
    
    st.dataframe(
        display_df.style.format({'correlation': '{:.3f}'}).background_gradient(subset=['correlation'], cmap='RdYlGn_r'),
        use_container_width=True,
        height=400
    )
    
    st.markdown("### Distribución de Correlaciones")
    
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.hist(result_df['correlation'], bins=30, alpha=0.7, color='steelblue', edgecolor='black')
    ax.axvline(result_df['correlation'].mean(), color='red', linestyle='--', linewidth=2, label='Promedio')
    ax.set_xlabel('Correlación')
    ax.set_ylabel('Frecuencia')
    ax.set_title('Distribución de Correlaciones')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    st.pyplot(fig, use_container_width=True)

# ========================================================================
# TAB 4: PORTFOLIO OPTIMIZER
# ========================================================================

with tab4:
    st.header("🎯 Portfolio Optimizer")
    
    optimization_method = st.selectbox(
        "Método de optimización:",
        ['uniform', 'inverse_dd', 'sharpe', 'sortino'],
    )
    
    weights = portfolio_allocation_minimum_dd(strategies_dict, method=optimization_method)
    
    st.markdown("### Asignación de Pesos Recomendada")
    
    weights_data = [{'Estrategia': k, 'Peso %': v*100} for k, v in weights.items()]
    weights_df = pd.DataFrame(weights_data).sort_values('Peso %', ascending=False)
    
    st.dataframe(
        weights_df.style.format({'Peso %': '{:.2f}%'}),
        use_container_width=True
    )
    
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.pie(
        weights_df['Peso %'],
        labels=weights_df['Estrategia'],
        autopct='%1.1f%%',
        startangle=90
    )
    plt.title(f'Asignación de Portafolio - {optimization_method.upper()}')
    st.pyplot(fig, use_container_width=True)
    
    st.markdown("### Métricas del Portafolio Combinado")
    
    portfolio_metrics = portfolio_combined_metrics(strategies_dict, weights)
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("PnL Total", f"${portfolio_metrics['total_pnl']:.2f}")
    with col2:
        st.metric("Max DD", f"${portfolio_metrics['max_dd']:.2f}")
    with col3:
        st.metric("Win Rate", f"{portfolio_metrics['win_rate']:.2f}%")
    with col4:
        st.metric("Sharpe", f"{portfolio_metrics['sharpe']:.2f}")

# ========================================================================
# TAB 5: ANÁLISIS DETALLADO
# ========================================================================

with tab5:
    st.header("📊 Análisis Detallado de Estrategias")
    
    selected_strat = st.selectbox(
        "Selecciona estrategia:",
        list(strategies_dict.keys())
    )
    
    df_selected = strategies_dict[selected_strat]
    pnl = df_selected['pnl'].fillna(0)
    equity = pnl.cumsum()
    
    st.markdown("### Curva de Equity")
    fig, ax = plt.subplots(figsize=(14, 6))
    ax.plot(equity.values, linewidth=2, label='Equity')
    ax.fill_between(range(len(equity)), equity.values, alpha=0.3)
    ax.set_title(f'Equity Curve - {selected_strat}')
    ax.set_xlabel('Trade #')
    ax.set_ylabel('Profit/Loss')
    ax.grid(True, alpha=0.3)
    ax.legend()
    st.pyplot(fig, use_container_width=True)
    
    st.markdown("### Drawdown Analysis")
    dd, dd_pct, running_max = calculate_drawdown_series(pnl)
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8))
    
    ax1.plot(equity.values, linewidth=2, label='Equity')
    ax1.plot(running_max.values, linewidth=2, linestyle='--', label='Running Max', alpha=0.7)
    ax1.fill_between(range(len(equity)), equity.values, running_max.values, alpha=0.2)
    ax1.set_title('Equity vs Peak')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    ax2.bar(range(len(dd_pct)), dd_pct.values, alpha=0.7, color='red')
    ax2.set_title('Drawdown %')
    ax2.set_ylabel('DD %')
    ax2.grid(True, alpha=0.3)
    
    st.pyplot(fig, use_container_width=True)

# ========================================================================
# TAB 6: INSIGHTS
# ========================================================================

with tab6:
    st.header("💡 Insights y Recomendaciones")
    
    corr_results, strat_names = calculate_dd_correlation_methods(strategies_dict)
    
    n = len(strat_names)
    min_corr = 2
    min_pair = None
    max_corr = -2
    max_pair = None
    
    for i in range(n):
        for j in range(i+1, n):
            if corr_results['pearson'][i, j] < min_corr:
                min_corr = corr_results['pearson'][i, j]
                min_pair = (strat_names[i], strat_names[j])
            if corr_results['pearson'][i, j] > max_corr:
                max_corr = corr_results['pearson'][i, j]
                max_pair = (strat_names[i], strat_names[j])
    
    st.markdown(f"""
    #### Pares de Estrategias:
    
    **Mejor Diversificación:**
    - {min_pair[0]} + {min_pair[1]}: {min_corr:.3f} ✅
    
    **Mayor Riesgo:**
    - {max_pair[0]} + {max_pair[1]}: {max_corr:.3f} ⚠️
    """)
    
    summary_data = []
    for strat_name, df in strategies_dict.items():
        metrics = calculate_strategy_metrics(df)
        summary_data.append(metrics)
    
    avg_pnl = np.mean([m['total_pnl'] for m in summary_data])
    avg_dd = np.mean([m['max_dd'] for m in summary_data])
    avg_sharpe = np.mean([m['sharpe'] for m in summary_data])
    
    st.markdown(f"""
    #### Estadísticas:
    - **PnL Promedio**: ${avg_pnl:.2f}
    - **DD Promedio**: ${avg_dd:.2f}
    - **Sharpe Promedio**: {avg_sharpe:.2f}
    """)
