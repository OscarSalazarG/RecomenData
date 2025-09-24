# RecomenData
Product Recomendation System applied in Retail company

<!--
  README.html — Executive, developer-grade summary for RecomenData.ipynb
  Author: Equipo Automation (Marca y Estrategia Comercial TLS)
  Purpose: GitHub-ready HTML summary of the notebook’s intent, pipeline, and key implementation details.
-->

<article style="font-family: ui-sans-serif, system-ui, -apple-system, Segoe UI, Roboto, Helvetica, Arial; line-height: 1.55; color: #0f172a;">
  <header>
    <h1 style="margin-bottom: 0.25rem;">Product Recommendation System for Mass-Retail Customers</h1>
    <p style="margin: 0; color:#475569;">
      End-to-end data preparation, reporting, and collaborative filtering (ALS/PySpark) modeling, extracted from
      <code>RecomenData.ipynb</code>.
    </p>
    <p style="margin-top: 0.5rem; color:#64748b;">
      Tech stack: <strong>Pandas, Seaborn/Matplotlib, scikit-learn (imputation/scaling), PySpark (ALS)</strong>.
    </p>
  </header>

  <section>
    <h2>1) Objectives</h2>
    <ul>
      <li><strong>Clean & harmonize</strong> a transactional dataset (2023) for retail purchases.</li>
      <li><strong>Produce descriptive reports</strong> (e.g., monthly sales, unique counts, basic profiling).</li>
      <li><strong>Generate operational assets</strong> for the web team:
        <ul>
          <li><code>usuarios.csv</code>: synthetic user credentials derived from unique customers.</li>
          <li><code>productos.csv</code>: unique product catalog with category descriptors.</li>
          <li><code>DataProcesada.csv</code>: model-ready dataset.</li>
        </ul>
      </li>
      <li><strong>Train a recommender system</strong> (Matrix Factorization via <em>Alternating Least Squares</em>) to suggest products per customer.</li>
    </ul>
  </section>

  <section>
    <h2>2) Data Ingestion & Renaming</h2>
    <p>The notebook reads a CSV transactional source and applies a consistent, Spanish-to-friendly column mapping.</p>
    <details open>
      <summary><strong>Column mapping</strong></summary>
      <table border="1" cellpadding="6" cellspacing="0">
        <thead style="background:#f1f5f9;">
          <tr>
            <th>Original</th><th>Renamed</th><th>Description (inferred)</th>
          </tr>
        </thead>
        <tbody>
          <tr><td><code>fec_documento</code></td><td><code>Fecha</code></td><td>Document date / transaction timestamp</td></tr>
          <tr><td><code>id_cliente_origen</code></td><td><code>ID_Cliente</code></td><td>Customer ID (source system)</td></tr>
          <tr><td><code>id_material_origen</code></td><td><code>ID_Producto</code></td><td>Product (SKU/material) ID</td></tr>
          <tr><td><code>cod_categoria</code></td><td><code>ID_Categoria</code></td><td>Category code</td></tr>
          <tr><td><code>des_categoria</code></td><td><code>Desc_Categoria</code></td><td>Category description</td></tr>
          <tr><td><code>tier_product</code></td><td><code>Calidad_Producto</code></td><td>Tier/quality tag (e.g., T0)</td></tr>
          <tr><td><code>des_fuerza_venta</code></td><td><code>FuerzaVenta</code></td><td>Sales force label</td></tr>
          <tr><td><code>ind_autoventa</code></td><td><code>Autoventa</code></td><td>Self-sale flag (Y/N)</td></tr>
          <tr><td><code>monto</code></td><td><code>Monto</code></td><td>Transaction amount/value</td></tr>
        </tbody>
      </table>
    </details>
  </section>

  <section>
    <h2>3) Data Cleaning, Types & Basic Imputation</h2>
    <ul>
      <li>Type casting for key IDs:
        <ul>
          <li><code>ID_Cliente</code>, <code>ID_Producto</code> → categorical during Pandas prep; cast to <code>int</code> in Spark phase.</li>
          <li><code>ID_Categoria</code> → <code>int</code> (missing as <code>0</code> pre-cast).</li>
        </ul>
      </li>
      <li>Missing handling:
        <ul>
          <li><code>Autoventa</code> → default <code>'N'</code></li>
          <li><code>Calidad_Producto</code> → default <code>'T0'</code></li>
          <li><code>FuerzaVenta</code> → default <code>'FFVV_0'</code></li>
          <li>Drop rows with missing <code>Desc_Categoria</code></li>
        </ul>
      </li>
      <li>Columns like <code>Unnamed: 0</code> are removed; duplicates are later handled for catalog extracts.</li>
    </ul>
    <p><em>Note:</em> While <code>KNNImputer</code>, <code>OrdinalEncoder</code>, and <code>MinMaxScaler</code> are imported, the modeling path relies on ALS with explicit ratings (<code>Monto</code>), so advanced feature scaling/encoding is not required in this first version.</p>
  </section>

  <section>
    <h2>4) Exploratory Data Analysis (EDA) & Business Checks</h2>
    <ul>
      <li><strong>Cardinality snapshot</strong>:
        <ul>
          <li>Unique <code>ID_Producto</code> (products)</li>
          <li>Unique <code>ID_Cliente</code> (customers)</li>
          <li>Unique <code>ID_Categoria</code> (categories)</li>
        </ul>
      </li>
      <li><strong>Descriptive statistics</strong> with <code>df.describe()</code> (floats formatted to 3 decimals).</li>
      <li><strong>Temporal aggregation</strong>:
        <ul>
          <li>Derives <code>Mes</code> from <code>Fecha</code> (<code>Period[M]</code> → timestamp), groups monthly sales, and visualizes with a line chart (Seaborn/Matplotlib).</li>
        </ul>
      </li>
    </ul>
    <pre style="background:#0b1020;color:#e2e8f0;padding:12px;border-radius:8px;overflow:auto;"><code># Monthly sales series (simplified)
df['Mes'] = df['Fecha'].dt.to_period('M')
ventas_por_mes = df.groupby('Mes')['Monto'].sum().reset_index()
ventas_por_mes['Mes'] = ventas_por_mes['Mes'].dt.to_timestamp()
sns.lineplot(x='Mes', y='Monto', data=ventas_por_mes, marker='o')</code></pre>
  </section>

  <section>
    <h2>5) Operational Exports for the Web Team</h2>
    <ul>
      <li><code>usuarios.csv</code> — One row per unique customer with synthetic credentials:
        <ul>
          <li><code>Usuario</code> and <code>Contraseña</code> = <em>Usuario&lt;ID_Cliente&gt;</em> (placeholder setup for demos/testing).</li>
        </ul>
      </li>
      <li><code>productos.csv</code> — Unique product catalog from <code>ID_Producto</code> with:
        <ul>
          <li><code>ID_Categoria</code>, <code>Desc_Categoria</code></li>
          <li>Duplicates removed by product key.</li>
        </ul>
      </li>
      <li><code>DataProcesada.csv</code> — Full, cleaned dataset saved for downstream Spark ingestion.</li>
    </ul>
  </section>

  <section>
    <h2>6) Recommendation Model (ALS on PySpark)</h2>
    <p><strong>Matrix Factorization</strong> via <em>Alternating Least Squares (ALS)</em> is used to learn latent factors for users and items from observed “ratings”. In this notebook, the monetary value <code>Monto</code> is treated as the implicit/explicit signal for preference strength.</p>

    <h3>6.1 Spark Session</h3>
    <pre style="background:#0b1020;color:#e2e8f0;padding:12px;border-radius:8px;overflow:auto;"><code>spark = (SparkSession.builder
  .appName("Recomendata")
  .config("spark.executor.memory", "5g")
  .config("spark.driver.memory", "5g")
  .getOrCreate())</code></pre>

    <h3>6.2 Data Schema for Modeling</h3>
    <p>After loading <code>DataProcesada.csv</code>, the modeling subset keeps only:</p>
    <ul>
      <li><code>ID_Cliente</code> → <code>int</code></li>
      <li><code>ID_Producto</code> → <code>int</code></li>
      <li><code>Monto</code> → <code>float</code></li>
    </ul>
    <pre style="background:#0b1020;color:#e2e8f0;padding:12px;border-radius:8px;overflow:auto;"><code>data_filtered = data.select('ID_Cliente','ID_Producto','Monto') \
  .withColumn('ID_Cliente', F.col('ID_Cliente').cast('int')) \
  .withColumn('ID_Producto', F.col('ID_Producto').cast('int')) \
  .withColumn('Monto', F.col('Monto').cast('float'))</code></pre>

    <h3>6.3 ALS Configuration</h3>
    <ul>
      <li><code>maxIter=5</code>, <code>regParam=0.01</code></li>
      <li>Columns: <code>userCol="ID_Cliente"</code>, <code>itemCol="ID_Producto"</code>, <code>ratingCol="Monto"</code></li>
      <li><code>coldStartStrategy="drop"</code> to avoid NaNs in predictions for unseen users/items.</li>
      <li>Simple retry mechanism for robustness in long-running Spark sessions.</li>
    </ul>
    <pre style="background:#0b1020;color:#e2e8f0;padding:12px;border-radius:8px;overflow:auto;"><code>als = ALS(
  maxIter=5,
  regParam=0.01,
  userCol="ID_Cliente",
  itemCol="ID_Producto",
  ratingCol="Monto",
  coldStartStrategy="drop"
)
model = als.fit(data_filtered)</code></pre>

    <h3>6.4 Inference</h3>
    <p>Top-N recommendations per user are computed with <code>recommendForAllUsers(k)</code>. The notebook prints recommendations for a sample <code>user_id = 4</code>.</p>
    <pre style="background:#0b1020;color:#e2e8f0;padding:12px;border-radius:8px;overflow:auto;"><code>user_recommendations = model.recommendForAllUsers(10)
user_id = 4
(
  user_recommendations
  .filter(F.col('ID_Cliente') == user_id)
  .show(truncate=False)
)</code></pre>

    <h3>6.5 Notes & Considerations</h3>
    <ul>
      <li><strong>Signal choice:</strong> Using <code>Monto</code> as a “rating” favors high-value purchases; consider normalization per user (e.g., z-score or log-transform) or use counts/implicit feedback (confidence weights) depending on business goals.</li>
      <li><strong>Cold start:</strong> New users/items require fallback strategies (e.g., popularity within <code>ID_Categoria</code>, content-based features, or nearest-neighbor warm-starts).</li>
      <li><strong>Evaluation:</strong> The current notebook focuses on training & generation. For production, add metrics such as <em>MAP@K</em>, <em>Precision@K</em>, or <em>NDCG@K</em> using temporal splits.</li>
    </ul>
  </section>

  <section>
    <h2>7) Pipeline at a Glance</h2>
    <ol>
      <li><strong>Read</strong> raw CSV → remove index artifacts → standardize schema.</li>
      <li><strong>Clean & cast</strong> key fields → fill strategic defaults → drop critical nulls.</li>
      <li><strong>EDA</strong>: counts, describes, monthly sales time-series.</li>
      <li><strong>Exports</strong>: <code>usuarios.csv</code>, <code>productos.csv</code>, <code>DataProcesada.csv</code>.</li>
      <li><strong>Spark</strong>: load processed CSV → cast to numeric → train ALS → compute Top-N recommendations.</li>
    </ol>
  </section>

  <section>
    <h2>8) Reproducibility & How to Run</h2>

    <h3>8.1 Minimal Environment</h3>
    <ul>
      <li>Python ≥ 3.10</li>
      <li>Packages:
        <code>pandas</code>, <code>numpy</code>, <code>matplotlib</code>, <code>seaborn</code>,
        <code>scikit-learn</code>, <code>pyspark</code>
      </li>
    </ul>

    <h3>8.2 Suggested <code>requirements.txt</code></h3>
    <pre style="background:#0b1020;color:#e2e8f0;padding:12px;border-radius:8px;overflow:auto;"><code>pandas>=2.2
numpy>=1.26
matplotlib>=3.8
seaborn>=0.13
scikit-learn>=1.4
pyspark>=3.5</code></pre>

    <h3>8.3 Local Execution (Notebook)</h3>
    <ol>
      <li>Place raw CSV under <code>Origen de datos/</code> (the notebook references a 2023 transactional CSV).</li>
      <li>Run all preprocessing cells until <strong>DataProcesada.csv</strong> is generated.</li>
      <li>Start Spark session and run ALS training cells.</li>
      <li>Optionally, persist <code>user_recommendations</code> to Parquet/CSV for downstream services.</li>
    </ol>

    <h3>8.4 Data Contracts (Expectations)</h3>
    <ul>
      <li><code>Fecha</code> must parse to datetime.</li>
      <li><code>ID_Cliente</code>, <code>ID_Producto</code> must be castable to integers.</li>
      <li><code>Monto</code> must be numeric (float), non-negative.</li>
      <li><code>Desc_Categoria</code> required (rows removed otherwise).</li>
    </ul>
  </section>

  <section>
    <h2>9) Extensibility & Next Steps</h2>
    <ul>
      <li><strong>Implicit Feedback ALS</strong> (set <code>implicitPrefs=True</code>) with confidence weighting on binary interactions (purchase/no purchase) or frequency.</li>
      <li><strong>Per-user normalization</strong> of <code>Monto</code> (e.g., log or min-max within user) to reduce bias toward high spenders.</li>
      <li><strong>Model selection</strong>:
        <ul>
          <li>Parameter sweeps for <code>rank</code>, <code>regParam</code>, <code>alpha</code> (implicit) with train/val splits.</li>
          <li>Temporal cross-validation for robust offline metrics (<em>MAP@K</em>, <em>NDCG@K</em>).</li>
        </ul>
      </li>
      <li><strong>Hybrid ranking</strong>:
        <ul>
          <li>Blend ALS scores with popularity within <code>ID_Categoria</code> and/or content similarity (cosine over product attributes).</li>
        </ul>
      </li>
      <li><strong>Cold-start strategies</strong>:
        <ul>
          <li>Trending products by category or recent sales.</li>
          <li>Ask-few-questions onboarding to seed initial vectors.</li>
        </ul>
      </li>
      <li><strong>MLOps/Prod</strong>:
        <ul>
          <li>Persist model via <code>model.write().overwrite().save(path)</code></li>
          <li>Batch scoring with Spark jobs; REST layer for online retrieval.</li>
          <li>Feature/metadata stores for <code>ID_Producto</code>, <code>ID_Categoria</code>, <code>Desc_Categoria</code>.</li>
        </ul>
      </li>
    </ul>
  </section>

  <section>
    <h2>10) File Outputs</h2>
    <table border="1" cellpadding="6" cellspacing="0">
      <thead style="background:#f1f5f9;">
        <tr><th>File</th><th>Shape/Keys</th><th>Purpose</th></tr>
      </thead>
      <tbody>
        <tr>
          <td><code>usuarios.csv</code></td>
          <td>Unique <code>ID_Cliente</code>; columns: <em>(ID_Cliente, Usuario, Contraseña)</em></td>
          <td>Web/demo auth scaffolding (synthetic). Replace with proper auth in prod.</td>
        </tr>
        <tr>
          <td><code>productos.csv</code></td>
          <td>Unique <code>ID_Producto</code>; columns: <em>(ID_Producto, ID_Categoria, Desc_Categoria)</em></td>
          <td>Product catalog for UI joins and display.</td>
        </tr>
        <tr>
          <td><code>DataProcesada.csv</code></td>
          <td>Transactional rows; cleaned schema</td>
          <td>Single source of truth for Spark training & analysis.</td>
        </tr>
      </tbody>
    </table>
  </section>

  <section>
    <h2>11) Key Snippets (Reference)</h2>
    <details>
      <summary><strong>Renaming & Basic Coercions</strong></summary>
      <pre style="background:#0b1020;color:#e2e8f0;padding:12px;border-radius:8px;overflow:auto;"><code>new_column_names = {
  'fec_documento': 'Fecha',
  'id_cliente_origen': 'ID_Cliente',
  'id_material_origen': 'ID_Producto',
  'cod_categoria': 'ID_Categoria',
  'des_categoria': 'Desc_Categoria',
  'tier_product': 'Calidad_Producto',
  'des_fuerza_venta': 'FuerzaVenta',
  'ind_autoventa': 'Autoventa',
  'monto': 'Monto'
}
df.rename(columns=new_column_names, inplace=True)

df['ID_Categoria'] = df['ID_Categoria'].fillna(0).astype(int)
df['Autoventa'] = df['Autoventa'].fillna('N')
df['Calidad_Producto'] = df['Calidad_Producto'].fillna('T0')
df['FuerzaVenta'] = df['FuerzaVenta'].fillna('FFVV_0')
df.dropna(subset=['Desc_Categoria'], inplace=True)</code></pre>
    </details>

    <details>
      <summary><strong>Unique Products Export</strong></summary>
      <pre style="background:#0b1020;color:#e2e8f0;padding:12px;border-radius:8px;overflow:auto;"><code>df_productos_unicos = (
  df[['ID_Producto','ID_Categoria','Desc_Categoria']]
  .drop_duplicates(subset='ID_Producto')
)
df_productos_unicos.to_csv('productos.csv', index=False)</code></pre>
    </details>

    <details>
      <summary><strong>Users Export</strong></summary>
      <pre style="background:#0b1020;color:#e2e8f0;padding:12px;border-radius:8px;overflow:auto;"><code>df_usuarios = pd.DataFrame({'ID_Cliente': df['ID_Cliente'].unique()})
df_usuarios['Usuario'] = 'Usuario' + df_usuarios['ID_Cliente'].astype(str)
df_usuarios['Contraseña'] = df_usuarios['Usuario']
df_usuarios.to_csv('usuarios.csv', index=False)</code></pre>
    </details>
  </section>

  <section>
    <h2>12) Governance & Quality Recommendations</h2>
    <ul>
      <li><strong>Schema evolution</strong>: lock incoming CSV schema with validation (e.g., <code>pandera</code>, <code>pydantic</code> or Great Expectations).</li>
      <li><strong>PII</strong>: ensure <code>ID_Cliente</code> is a surrogate key; keep raw PII compliant and outside analytics outputs.</li>
      <li><strong>Reproducibility</strong>: pin versions in <code>requirements.txt</code> and freeze model env with Docker/Conda.</li>
      <li><strong>Monitoring</strong>: track coverage (% users with recommendations), sparsity, and drift in basket composition.</li>
    </ul>
  </section>

  <footer style="margin-top:2rem; border-top: 1px solid #e2e8f0; padding-top: 1rem; color:#64748b;">
    <p>
      <strong>Notebook:</strong> <code>RecomenData.ipynb</code> •
      <strong>Primary outputs:</strong> <code>usuarios.csv</code>, <code>productos.csv</code>, <code>DataProcesada.csv</code> •
      <strong>Model:</strong> PySpark ALS (Top-N recommendations).
    </p>
    <p>For productionization, consider a batch scoring job (Spark) plus a lightweight API to serve Top-N by <code>ID_Cliente</code>, enriched with <code>productos.csv</code> for title/description rendering.</p>
  </footer>
</article>
