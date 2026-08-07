

# Decodificación Especulativa Jerárquica (HSD)

**Decodificación Especulativa Jerárquica (HSD)** es un algoritmo de verificación sin pérdida de última generación para acelerar la inferencia de modelos de lenguaje grandes (LLM). HSD verifica las salidas de borrador utilizando la probabilidad conjunta de múltiples tokens de borrador y reconstruye la distribución conjunta del modelo objetivo a través de una jerarquía de distribuciones de remuestreo en diferentes ramas. Esta verificación jerárquica permite que HSD **acepte más tokens en promedio** que los métodos de verificación token por token y por bloques, al tiempo que **preserva estrictamente la distribución de salida del modelo objetivo**, es decir, sin ninguna degradación en el rendimiento del modelo.

Además, su fuerte explicabilidad y generalidad facilitan su integración en una amplia variedad de marcos de decodificación especulativa. Cabe destacar que HSD actúa como un **reemplazo directo** para la verificación token por token. Por ejemplo, al reemplazar directamente la verificación token por token en un modelo EAGLE-3 pre-ajustado, HSD logra una mejora de rendimiento superior al 12% **sin requerir ningún ajuste fino adicional**. 

## Índice

- [Decodificación Especulativa Jerárquica (HSD)](#hierarchical-speculative-decoding-hsd)
  - [Índice](#table-of-contents)
  - [Configuración de los Experimentos](#experiment-settings)
    - [Configuración de los Experimentos](#experiments-setup)
    - [Líneas base y métricas](#baselines-and-metrics)
  - [Reproducir los resultados de los experimentos en GSM8K](#reproduce-experiment-results-on-gsm8k)
    - [Requisitos](#requirements)
    - [Verificación de un solo borrador](#single-draft-verification)
    - [Verificación de múltiples borradores con muestreo de rechazo recursivo](#multi-draft-verification-with-recursive-reject-sampling)
    - [Cálculo de métricas](#compute-metrics)
    - [Resultados de rendimiento en GSM8K](#gsm8k-performance-results)
      - [Comparación de precisión (Tabla 1)](#accuracy-comparison-table-1)
      - [Eficiencia de bloque y velocidad (Tabla 2)](#block-efficiency--speed-table-2)
  - [Integración con EAGLE](#eagle-integration)
    - [Resultados de EAGLE + HSD](#eagle--hsd-results)
  - [Cita](#citation)
  - [Referencias](#references)

## Configuración de los Experimentos
### Configuración de los Experimentos
Los experimentos se realizan con la serie Qwen2.5 instruida y cuantizada con GPTQ en 8 bits, ampliamente adoptada [Bai et al., 2023](#). De forma predeterminada, utilizamos el modelo de 0.5B como modelo de borrador y el de 72B como modelo objetivo, con una temperatura de 1. Todos los experimentos se llevaron a cabo en una sola GPU NVIDIA H20 con 96 GB de memoria, a menos que se indique lo contrario.

### Líneas base y métricas
Comparamos dos métodos de verificación sin pérdida: Token-wise [Leviathan et al., 2023](https://arxiv.org/abs/2211.17192) y Block-wise [Sun et al., 2024](https://arxiv.org/abs/2403.10444), utilizando dos métricas: *Eficiencia de Bloque* (tokens/paso) y *Velocidad de Decodificación* (tokens/segundo).
- **Eficiencia de Bloque** mide el número promedio de tokens generados por cada llamada serial al modelo objetivo, reflejando una eficiencia intrínseca independiente del hardware.  
- **Velocidad de Decodificación** indica los tokens producidos por segundo para referencia práctica, aunque depende del hardware y la implementación.


## Reproducir los resultados de los experimentos en GSM8K

### Requisitos
Instale `python=3.10` y `transformers=4.46.3`, y luego copie los archivos proporcionados dentro del directorio `transformers` a:

```
anaconda3/envs/your_environment_name/lib/python3.10/site-packages/transformers/
```
A continuación, navegue a `chain-of-thought-hub/gsm8k`:
```bash
cd chain-of-thought-hub/gsm8k
```

### Verificación de un solo borrador

Métodos base:
- Verificación **Tokenwise**
  ```bash
  bash eval_speculative_qwen.sh
  ```

Nuestros métodos:

- **HSD (clever)** - optimizado con un mecanismo de acotamiento inteligente
  ```bash
  bash eval_speculative_qwen_backward_clever.sh
  ```

### Verificación de múltiples borradores con muestreo de rechazo recursivo

Extiende HSD para utilizar múltiples borradores muestreados independientemente, logrando un rendimiento aún mejor.

- **Tokenwise** multidraft
  ```bash
  bash eval_speculative_qwen_multidraft_11.sh
  ```

- **HSD** multidraft
  ```bash
  bash eval_speculative_qwen_backward_clever_multidraft_11.sh
  ```

### Cálculo de métricas

Finalmente, evalúe la Eficiencia de Bloque y la Velocidad de Decodificación:
```bash
python compute_speculative_stats.py
```

### Resultados de rendimiento en GSM8K

Utilizando Qwen2.5-0.5B como modelo de borrador en GSM8K:

#### **Comparación de precisión (Tabla 1)](#accuracy-comparison-table-1)
<table>
  <thead>
    <tr>
      <th>Modelo Objetivo</th>
      <th>Tokenwise</th>
      <th>HSD (nuestro)</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>Qwen2.5-72B</td>
      <td>0.8213</td>
      <td><strong>0.8517</strong></td>
    </tr>
    <tr>
      <td>Qwen2.5-32B</td>
      <td>0.8213</td>
      <td><strong>0.8479</strong></td>
    </tr>
    <tr>
      <td>Qwen2.5-14B</td>
      <td>0.8327</td>
      <td><strong>0.8327</strong></td>
    </tr>
  </tbody>
</table>

HSD mantiene o mejora ligeramente la precisión (ambos métodos, Token-wise y HSD, son sin pérdida, por lo que cualquier desviación se debe a la aleatoriedad), al tiempo que ofrece una inferencia significativamente más rápida.

#### **Eficiencia de bloque y velocidad (Tabla 2)](#block-efficiency--speed-table-2) 
| Método         | EB 14B | EB 32B | EB 72B | VD 14B | VD 32B | VD 72B |
|----------------|--------|--------|--------|--------|--------|--------|
| Tokenwise      | 5.99   | 6.14   | 6.44   | 82.28  | 53.87  | 31.49  |
| Blockwise      | 6.13 (+2.3%) | 6.26 (+2.0%) | 6.53 (+1.4%) | 86.06 (+4.6%) | 54.91 (+1.9%) | 31.79 (+1.0%) |
| **HSD (Nuestro)** | **6.30 (+5.2%)** | **6.47 (+5.4%)** | **6.65 (+3.3%)** | **91.05 (+10.7%)** | **57.12 (+6.0%)** | **32.52 (+3.3%)** |


HSD mejora consistentemente tanto la Eficiencia de Bloque (EB) como la Velocidad de Decodificación (VD) en comparación con la verificación Tokenwise y Blockwise. Para **GSM8K**, las ganancias son estables en todas las escalas, con mejoras en la EB de **5.2%-5.4%** en 14B/32B y **3.3%** en 72B, acompañadas de aumentos en la VD de hasta **10.7%**.

## Integración con EAGLE

HSD puede integrarse con [EAGLE-3](https://github.com/SforAiD/LLM-Eagle) para mejorar aún más el rendimiento. EAGLE-3 ajusta finamente un componente adicional para predecir tokens de borrador a partir de las características ocultas del modelo objetivo, eliminando la necesidad de un modelo de borrador separado y, por lo tanto, mejorando la velocidad de decodificación. 

Para demostrar la compatibilidad de HSD con sistemas de decodificación avanzados, lo integramos en el modelo de última generación **EAGLE-3-LLaMa3.1-Instruct-8B** (con una longitud de borrador predeterminada de 7) reemplazando su verificador token-wise, como se muestra en la **Tabla 3**. Es importante destacar que HSD opera de manera ortogonal a la fase de redacción, por lo que puede utilizarse directamente con un modelo EAGLE-3 pre-ajustado **sin requerir ningún ajuste fino adicional**.

### **Tabla 3**: Resultados de EAGLE-3 + HSD (EAGLE-3H)
| Método               | Eficiencia de Bloque | Velocidad de Decodificación |
|----------------------|----------------------|-----------------------------|
| EAGLE-3              | 3.22                 | 116.50                      |
| Blockwise            | N/A                  | N/A                         |
| **EAGLE-3H (Nuestro)**  | **3.61 (+12.1%)**    | **135.42 (+16.2%)**         |


Para configurar la integración con EAGLE, consulte el directorio `EAGLE-hsd/` en este repositorio.

```bash
cd EAGLE-3H
bash eagle_eval.sh
```

## Cita

```bibtex
@misc{zhou2026overcomingjointintractabilitylossless,
      title={Overcoming Joint Intractability with Lossless Hierarchical Speculative Decoding}, 
      author={Yuxuan Zhou and Fei Huang and Heng Li and Fengyi Wu and Tianyu Wang and Jianwei Zhang and Junyang Lin and Zhi-Qi Cheng},
      year={2026},
      eprint={2601.05724},
      archivePrefix={arXiv},
      primaryClass={cs.AI},
      url={https://arxiv.org/abs/2601.05724}, 
}
```

## Referencias

1. [Chain-of-Thought Hub](https://github.com/FranxYao/chain-of-thought-hub)  
2. [EAGLE-3 Repository](https://github.com/SforAiD/LLM-Eagle)
