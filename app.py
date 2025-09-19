# -*- coding: utf-8 -*-
# =============================================================================
#         CALCULADORA DE REGRESSÃO LINEAR COM STREAMLIT
# =============================================================================
import streamlit as st
import pandas as pd
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt
import seaborn as sns
import io
import base64

# --- CONFIGURAÇÕES DA PÁGINA E ESTILOS ---
st.set_page_config(layout="wide", page_title="Análise de Regressão Linear")

# --- INFORMAÇÕES DO AUTOR E TÍTULO ---
st.markdown("Elaborado por Tales Rabelo Freitas")
st.markdown("LinkedIn: [https://www.linkedin.com/in/tales-rabelo-freitas-1a1466187/](https://www.linkedin.com/in/tales-rabelo-freitas-1a1466187/)")
st.title("📊 Análise de Regressão Linear")
st.markdown("Faça o upload de uma planilha Excel, selecione suas variáveis e obtenha uma análise de regressão completa.")


sns.set_theme(style="whitegrid")
plt.rcParams['figure.figsize'] = (10, 5)

# --- FUNÇÕES AUXILIARES ---
def gerar_relatorio_html(y_var, x_vars, df_corr, fig, results_summary, equation, messages):
    img_buf = io.BytesIO()
    fig.savefig(img_buf, format='png', bbox_inches='tight')
    img_b64 = base64.b64encode(img_buf.getvalue()).decode('utf-8')
    img_buf.close()
    corr_html = df_corr.style.background_gradient(cmap='coolwarm').format("{:.2f}").to_html()
    summary_html = results_summary.as_html()
    messages_html = "".join([f"<p><em>{msg}</em></p>" for msg in messages])
    html_content = f'''
    <html>
    <head>
        <meta charset="UTF-8"><title>Relatório de Regressão</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 40px; color: #333; }}
            h1, h2 {{ color: #005b96; border-bottom: 2px solid #005b96; padding-bottom: 5px; }}
            .section {{ margin-bottom: 40px; padding: 15px; border: 1px solid #ddd; border-radius: 5px; background-color: #f9f9f9; overflow-x: auto; }}
            pre {{ background-color: #eee; padding: 15px; border: 1px solid #ccc; border-radius: 5px; white-space: pre-wrap; word-wrap: break-word; }}
            img {{ max-width: 100%; height: auto; border: 1px solid #ddd; border-radius: 5px; margin-top: 10px; }}
            table {{ border-collapse: collapse; margin: 15px 0; }}
            th, td {{ border: 1px solid #ccc; padding: 8px; text-align: center; }}
            th {{ background-color: #e9ecef; }}
        </style>
    </head>
    <body>
        <h1>Relatório de Análise de Regressão</h1>
        <div class="section">
            <h2>Configuração da Análise</h2>
            <p><strong>Variável Dependente (Y):</strong> {y_var}</p>
            <p><strong>Variáveis Explicativas (X):</strong> {', '.join(x_vars)}</p>
            {messages_html}
        </div>
        <div class="section"><h2>1. Matriz de Correlação</h2>{corr_html}</div>
        <div class="section"><h2>2. Gráficos de Dispersão</h2><img src="data:image/png;base64,{img_b64}" alt="Gráficos de Dispersão"></div>
        <div class="section"><h2>3. Sumário da Regressão</h2>{summary_html}</div>
        <div class="section"><h2>4. Equação da Regressão</h2><pre>{equation}</pre></div>
    </body>
    </html>
    '''
    return html_content

# --- INTERFACE DO STREAMLIT ---
st.sidebar.header("Configurações da Análise")
uploaded_file = st.sidebar.file_uploader("1. Escolha sua planilha Excel", type=["xlsx", "xls"])

if uploaded_file:
    try:
        df = pd.read_excel(uploaded_file)
        st.subheader("Amostra dos Dados Carregados")
        st.dataframe(df.head())
        column_list = df.columns.tolist()
        y_var_name = st.sidebar.selectbox("2. Selecione a Variável Dependente (Y)", options=column_list, index=None, placeholder="Escolha uma opção")
        x_vars_names = st.sidebar.multiselect("3. Selecione as Variáveis Explicativas (X)", options=[col for col in column_list if col != y_var_name], placeholder="Escolha uma ou mais opções")
        remove_outliers = st.sidebar.checkbox("Remover Outliers (Método IQR)", value=False)
        if st.sidebar.button("Calcular Regressão", type="primary"):
            if not y_var_name or not x_vars_names:
                st.error("Erro: Por favor, selecione a variável Y e pelo menos uma variável X.")
            elif y_var_name in x_vars_names:
                st.error(f"Erro: A variável dependente '{y_var_name}' não pode estar na lista de variáveis explicativas.")
            else:
                with st.spinner("Calculando... Por favor, aguarde."):
                    report_messages = []
                    selected_cols = [y_var_name] + x_vars_names
                    analysis_df = df[selected_cols].copy()
                    initial_rows = len(analysis_df)
                    analysis_df.dropna(inplace=True)
                    if len(analysis_df) < initial_rows:
                        msg = f"Aviso: {initial_rows - len(analysis_df)} linha(s) com dados ausentes foram removidas."
                        st.info(msg)
                        report_messages.append(msg)
                    if remove_outliers:
                        rows_before = len(analysis_df)
                        outlier_indices = set()
                        for col in selected_cols:
                            Q1, Q3 = analysis_df[col].quantile(0.25), analysis_df[col].quantile(0.75)
                            IQR = Q3 - Q1
                            lower, upper = Q1 - 1.5 * IQR, Q3 + 1.5 * IQR
                            col_outliers = analysis_df[(analysis_df[col] < lower) | (analysis_df[col] > upper)].index
                            outlier_indices.update(col_outliers)
                        analysis_df.drop(index=list(outlier_indices), inplace=True)
                        num_removed = rows_before - len(analysis_df)
                        msg = f"{num_removed} outlier(s) foram identificados e removidos." if num_removed > 0 else "Nenhum outlier encontrado."
                        st.info(msg)
                        report_messages.append(msg)
                    if len(analysis_df) < len(x_vars_names) + 2:
                        st.error("Erro: Não há dados suficientes para realizar a análise após a limpeza dos dados.")
                    else:
                        st.success(f"Análise final será feita com **{len(analysis_df)}** observações.")
                        st.header("1. Matriz de Correlação")
                        correlation_matrix = analysis_df.corr()
                        st.dataframe(correlation_matrix.style.background_gradient(cmap='coolwarm').format("{:.2f}"))
                        st.header("2. Gráficos de Dispersão (Y vs. X)")
                        num_x_vars = len(x_vars_names)
                        cols = st.columns(min(num_x_vars, 2))
                        for i, x_var in enumerate(x_vars_names):
                            fig, ax = plt.subplots()
                            sns.regplot(x=x_var, y=y_var_name, data=analysis_df, ax=ax, line_kws={"color": "red"}, scatter_kws={'alpha': 0.6})
                            ax.set_title(f'Relação entre {y_var_name} e {x_var}')
                            cols[i % 2].pyplot(fig)
                        st.header("3. Resultados do Modelo de Regressão")
                        Y = analysis_df[y_var_name]
                        X = sm.add_constant(analysis_df[x_vars_names])
                        results = sm.OLS(Y, X).fit()
                        st.text(results.summary())
                        st.header("4. Equação da Regressão")
                        equation = f"{y_var_name} = {results.params['const']:.4f}"
                        for var in x_vars_names: equation += f" + ({results.params[var]:.4f} * {var})"
                        equation_clean = equation.replace("+ (-", "- (")
                        st.code(equation_clean, language='text')
                        st.header("5. Exportar Relatório")
                        fig_report, axes = plt.subplots(nrows=(num_x_vars + 1) // 2, ncols=2, figsize=(14, 5 * ((num_x_vars + 1) // 2)))
                        axes_flat = axes.flatten()
                        for i, x_var in enumerate(x_vars_names):
                            sns.regplot(x=x_var, y=y_var_name, data=analysis_df, ax=axes_flat[i], line_kws={"color": "red"}, scatter_kws={'alpha': 0.6})
                            axes_flat[i].set_title(f'Relação entre {y_var_name} e {x_var}')
                        for i in range(num_x_vars, len(axes_flat)): axes_flat[i].set_visible(False)
                        plt.tight_layout()
                        html_report = gerar_relatorio_html(y_var_name, x_vars_names, correlation_matrix, fig_report, results.summary(), equation_clean, report_messages)
                        st.download_button(label="Baixar Relatório Completo (HTML)", data=html_report, file_name="relatorio_regressao.html", mime="text/html")
    except Exception as e:
        st.error(f"Ocorreu um erro ao processar o arquivo: {e}")
else:
    st.info("Aguardando o upload de um arquivo Excel para iniciar a análise.")
"""

# --- Conteúdo do arquivo requirements.txt ---
requirements_txt_code = """
streamlit
pandas
numpy
statsmodels
matplotlib
seaborn
openpyxl
"""

# --- Lógica para criar e baixar os arquivos ---
try:
    with open('app.py', 'w', encoding='utf-8') as f:
        f.write(app_py_code)
    print("✅ Arquivo 'app.py' atualizado com sucesso no ambiente do Colab.")

    with open('requirements.txt', 'w', encoding='utf-8') as f:
        f.write(requirements_txt_code)
    print("✅ Arquivo 'requirements.txt' criado com sucesso no ambiente do Colab.")

    print("\\nIniciando o download dos arquivos...")
    files.download('app.py')
    files.download('requirements.txt')
    print("\\n🚀 Download concluído! Agora você pode subir estes arquivos para o seu GitHub.")

except Exception as e:
    print(f"❌ Ocorreu um erro: {e}")
