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
st.markdown("**Análisis de correlación de estrategias basado en sincronización de drawdowns**")

# ========================================================================
# FUNCIONES DE CÁLCULO
# ========================================================================

def parse_sqx_csv(uploaded_file):
    """
    Lee CSV exportado directamente de SQX
    """
    try:
        # Intentar múltiples separadores
        for sep in [';', ',', '\t']:
            try:
                df = pd.read_csv(uploaded_file, sep=sep, encoding='utf-8')
                if len(df.columns) > 3:
                    break
            except:
                uploaded_file.seek(0)
                continue
        
        # Limpiar nombres de columnas
        df.columns = df.columns.str.strip().str.replace('"', '').str.lower()
        
        # Renombrar columnas esperadas
        rename_map = {
            'open time': 'open_time',
            'open price': 'open_price',
            'close time': 'close_time',
            'close price': 'close_price',
            'profit/loss': 'pnl',
            'size': 'volume',
        }
        df = df.rename(columns=rename_map)
        
        # Convertir tipos
        df['open_time'] = pd.to_datetime(df['open_time'], errors='coerce')
        df['close_time'] = pd.to_datetime(df['close_time'], errors='coerce')
        df['pnl'] = pd.to_numeric(df['pnl'], errors='coerce')
        df['balance'] = pd.to_numeric(df['balance'], errors='coerce')
        
        # Ordenar por tiempo
        df = df.sort_values('close_time').reset_index(drop=True)
        
        return df[['open_time', 'close_time', 'symbol', 'type', 'volume', 
                  'open_price', 'close_price', 'pnl', 'balance', 'comment']].copy()
    
    except Exception as e:
        st.error(f"Error parsing CSV: {e}")
        return None

def calculate_drawdown_series(pnl_series):
    """
    Calcula la serie de drawdown para una estrategia
    """
    cumulative = pnl_series.cumsum()
    running_max = cumulative.cummax()
    drawdown = running_max - cumulative
    
    # Normalizado (% del peak)
    drawdown_pct = (drawdown / running_max.replace(0, np.nan)) * 100
    drawdown_pct = drawdown_pct.fillna(0)
    
    return drawdown, drawdown_pct, running_max

def calculate_strategy_metrics(df):
    """Calcula métricas principales de una estrategia"""
    pnl = df['pnl'].fillna(0)
    
    total_pnl = pnl.sum()
    num_trades = len(pnl)
    win_rate = (pnl > 0).sum() / num_trades * 100 if num_trades > 0 else 0
    
    gains = pnl[pnl > 0].sum()
    losses = abs(pnl[pnl < 0].sum())
    pf = gains / losses if losses > 0 else (np.inf if gains > 0 else 0)
    
    # Drawdown
    cumulative = pnl.cumsum()
    running_max = cumulative.cummax()
    max_dd = (running_max - cumulative).max()
    
    # Sharpe (trades)
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
    """
    VERSIÓN CORREGIDA: Maneja diferentes longitudes de series
    """
    
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
    
    # Calcular series de drawdown para cada estrategia
    for strat_name, df in strategies_dict.items():
        pnl = df['pnl'].fillna(0).values
        dd, dd_pct, _ = calculate_drawdown_series(pd.Series(pnl))
        results['drawdown_series'][strat_name] = dd.values
        results['drawdown_pct'][strat_name] = dd_pct.values
    
    # Calcular correlaciones entre pares
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
                
                # ✅ CORRECCIÓN: Alinear series de diferentes longitudes
                min_len = min(len(dd1), len(dd2))
                dd1_aligned = dd1[:min_len]
                dd2_aligned = dd2[:min_len]
                
                # Método 1: Correlación Pearson de series de DD
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
                
                # Método 2: Ratio de sincronización
                in_dd1 = dd1_aligned > 0.5
                in_dd2 = dd2_aligned > 0.5
                simultaneous_dd = (in_dd1 & in_dd2).sum()
                any_dd = (in_dd1 | in_dd2).sum()
                sync_ratio = simultaneous_dd / any_dd if any_dd > 0 else 0
                results['sync_ratio'][i, j] = sync_ratio
                
                # Método 3: % tiempo en DD conjunto
                joint_dd = (in_dd1 & in_dd2).sum() / min_len * 100 if min_len > 0 else 0
                results['joint_dd_pct'][i, j] = joint_dd
                
                # Método 4: Correlación de timing de picos DD
                dd1_peaks = (dd1_aligned > np.roll(dd1_aligned, 1)) & (dd1_aligned > np.roll(dd1_aligned, -1))
                dd2_peaks = (dd2_aligned > np.roll(dd2_aligned, 1)) & (dd2_aligned > np.roll(dd2_aligned, -1))
                shared_peaks = (dd1_peaks & dd2_peaks).sum()
                all_peaks = (dd1_peaks | dd2_peaks).sum()
                timing_corr = shared_peaks / all_peaks if all_peaks > 0 else 0
                results['dd_timing'][i, j] = timing_corr
    
    return results, strategy_names

def portfolio_allocation_minimum_dd(strategies_dict, method='uniform'):
    """
    Sugiere asignación de portafolio basada en minimizar DD agregado
    """
    
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
    """
    Calcula métricas del portafolio combinado
    """
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
st.sidebar.markdown("Sube múltiples CSV de SQX para análisis de portafolio")

uploaded_files = st.sidebar.file_uploader(
    "📤 Selecciona archivos CSV (SQX export)",
    type=['csv'],
    accept_multiple_files=True
)

if not uploaded_files:
    st.info("⏳ Carga al menos 2 archivos CSV de estrategias para comenzar")
    st.stop()

# Parsear archivos
strategies_dict = {}
for uploaded_file in uploaded_files:
    df = parse_sqx_csv(uploaded_file)
    if df is not None and not df.empty:
        strat_name = uploaded_file.name.replace('.csv', '').replace('.CSV', '')
        strategies_dict[strat_name] = df

if len(strategies_dict) < 2:
    st.error("❌ Se necesitan al menos 2 estrategias válidas para análisis de portafolio")
    st.stop()

st.success(f"✅ {len(strategies_dict)} estrategias cargadas correctamente")

# ========================================================================
# TAB 1: RESUMEN DE ESTRATEGIAS
# ========================================================================

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📊 Resumen", 
    "🔗 Correlación DD", 
    "🎯 Portfolio Optimizer",
    "📈 Análisis Detallado",
    "💡 Insights"
])

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
    
    # Gráficos de resumen
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
# TAB 2: CORRELACIÓN DE DRAWDOWN
# ========================================================================

with tab2:
    st.header("🔗 Análisis de Correlación por Drawdown")
    
    # Calcular correlaciones
    corr_results, strat_names = calculate_dd_correlation_methods(strategies_dict)
    
    st.markdown("""
    ### Métodos de Correlación Disponibles:
    - **Pearson**: Correlación lineal de series de DD
    - **Spearman**: Correlación no-paramétrica (ranks)
    - **Sync Ratio**: % de tiempo en DD sincronizadas
    - **Joint DD %**: % tiempo ambas en drawdown simultáneamente
    - **DD Timing**: Correlación de picos de DD
    """)
    
    # Selector de método
    corr_method = st.selectbox(
        "Selecciona método de correlación:",
        ['pearson', 'spearman', 'sync_ratio', 'joint_dd_pct', 'dd_timing'],
        help="Diferentes perspectivas del riesgo correlacionado"
    )
    
    # Obtener matriz seleccionada
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
    
    # Heatmap
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
    plt.yticks(rotation=0)
    st.pyplot(fig, use_container_width=True)
    
    # Matriz como tabla
    st.markdown("### Matriz de Correlación")
    corr_df = pd.DataFrame(corr_matrix, index=strat_names, columns=strat_names)
    st.dataframe(
        corr_df.style.format('{:.3f}').background_gradient(cmap='RdYlGn_r'),
        use_container_width=True
    )
    
    # Análisis de pares
    st.markdown("### Análisis de Pares (Correlación más baja = mejor diversificación)")
    
    pairs_data = []
    for i, s1 in enumerate(strat_names):
        for j, s2 in enumerate(strat_names):
            if i < j:
                pairs_data.append({
                    'Estrategia 1': s1,
                    'Estrategia 2': s2,
                    'Pearson': corr_results['pearson'][i, j],
                    'Spearman': corr_results['spearman'][i, j],
                    'Sync Ratio': corr_results['sync_ratio'][i, j],
                    'Joint DD %': corr_results['joint_dd_pct'][i, j],
                    'DD Timing': corr_results['dd_timing'][i, j],
                })
    
    pairs_df = pd.DataFrame(pairs_data).sort_values('Pearson')
    st.dataframe(
        pairs_df.style.format({col: '{:.3f}' for col in pairs_df.columns if col != 'Estrategia 1' and col != 'Estrategia 2'}),
        use_container_width=True,
        height=400
    )

# ========================================================================
# TAB 3: PORTFOLIO OPTIMIZER
# ========================================================================

with tab3:
    st.header("🎯 Portfolio Optimizer")
    
    optimization_method = st.selectbox(
        "Método de optimización:",
        ['uniform', 'inverse_dd', 'sharpe', 'sortino'],
        help="""
        - Uniform: Igual peso para todas
        - Inverse DD: Más peso a estrategias con menor DD
        - Sharpe: Proporcional a Sharpe ratio
        - Sortino: Proporcional a Sortino ratio
        """
    )
    
    # Calcular pesos
    weights = portfolio_allocation_minimum_dd(strategies_dict, method=optimization_method)
    
    st.markdown("### Asignación de Pesos Recomendada")
    
    weights_data = [{'Estrategia': k, 'Peso %': v*100} for k, v in weights.items()]
    weights_df = pd.DataFrame(weights_data).sort_values('Peso %', ascending=False)
    
    st.dataframe(
        weights_df.style.format({'Peso %': '{:.2f}%'}),
        use_container_width=True
    )
    
    # Visualizar asignación
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.pie(
        weights_df['Peso %'],
        labels=weights_df['Estrategia'],
        autopct='%1.1f%%',
        startangle=90
    )
    plt.title(f'Asignación de Portafolio - {optimization_method.upper()}')
    st.pyplot(fig, use_container_width=True)
    
    # Métricas del portafolio
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
# TAB 4: ANÁLISIS DETALLADO
# ========================================================================

with tab4:
    st.header("📈 Análisis Detallado de Estrategias")
    
    selected_strat = st.selectbox(
        "Selecciona estrategia para análisis:",
        list(strategies_dict.keys())
    )
    
    df_selected = strategies_dict[selected_strat]
    
    # Equity curve
    st.markdown("### Curva de Equity")
    pnl = df_selected['pnl'].fillna(0)
    equity = pnl.cumsum()
    
    fig, ax = plt.subplots(figsize=(14, 6))
    ax.plot(equity.values, linewidth=2, label='Equity')
    ax.fill_between(range(len(equity)), equity.values, alpha=0.3)
    ax.set_title(f'Equity Curve - {selected_strat}')
    ax.set_xlabel('Trade #')
    ax.set_ylabel('Profit/Loss')
    ax.grid(True, alpha=0.3)
    ax.legend()
    st.pyplot(fig, use_container_width=True)
    
    # Drawdown chart
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
# TAB 5: INSIGHTS
# ========================================================================

with tab5:
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
    
    **Mejor Diversificación (menor correlación):**
    - {min_pair[0]} + {min_pair[1]}: Correlación = {min_corr:.3f}
    - ✅ Recomendado para combinar
    
    **Mayor Riesgo de Concentración (mayor correlación):**
    - {max_pair[0]} + {max_pair[1]}: Correlación = {max_corr:.3f}
    - ⚠️ Considerar reducir peso de una
    """)
    
    summary_data = []
    for strat_name, df in strategies_dict.items():
        metrics = calculate_strategy_metrics(df)
        summary_data.append(metrics)
    
    avg_pnl = np.mean([m['total_pnl'] for m in summary_data])
    avg_dd = np.mean([m['max_dd'] for m in summary_data])
    avg_sharpe = np.mean([m['sharpe'] for m in summary_data])
    
    st.markdown(f"""
    #### Estadísticas de Portafolio:
    
    - **PnL Promedio**: ${avg_pnl:.2f}
    - **DD Promedio**: ${avg_dd:.2f}
    - **Sharpe Promedio**: {avg_sharpe:.2f}
    """)
    
    st.markdown("#### Recomendaciones:")
    
    recommendations = []
    
    max_dd_strat = max(strategies_dict.keys(), key=lambda x: calculate_strategy_metrics(strategies_dict[x])['max_dd'])
    max_dd_val = calculate_strategy_metrics(strategies_dict[max_dd_strat])['max_dd']
    
    if max_dd_val > avg_dd * 1.5:
        recommendations.append(f"⚠️ **{max_dd_strat}** tiene DD muy alto ({max_dd_val:.2f}). Considerar reducir peso.")
    
    sharpes = {name: calculate_strategy_metrics(strategies_dict[name])['sharpe'] for name in strategies_dict.keys()}
    best_sharpe = max(sharpes, key=sharpes.get)
    
    recommendations.append(f"✅ **{best_sharpe}** tiene mejor Sharpe. Considerar aumentar peso.")
    
    high_corr_pairs = sum(1 for i in range(n) for j in range(i+1, n) if corr_results['pearson'][i, j] > 0.7)
    
    if high_corr_pairs > 0:
        recommendations.append(f"⚠️ Existen {high_corr_pairs} pares altamente correlacionados. Revisar para evitar concentración de riesgo.")
    else:
        recommendations.append(f"✅ Baja correlación general entre estrategias. Buena diversificación.")
    
    for rec in recommendations:
        st.markdown(rec)
