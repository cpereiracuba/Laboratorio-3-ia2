# Guía de Buenas Prácticas: Entrenamiento de Redes Neuronales (Optimización + Regularización)

> Cuadernillo unificado a partir de los 3 documentos originales: *receta de entrenamiento*, *optimización* y *regularización*. Pensado para dárselo a Claude como contexto de referencia en próximos laboratorios, sin tener que adjuntar los tres por separado.

---

## 0. Cómo usar esta guía

Este documento resume el flujo completo recomendado para entrenar una red neuronal de forma efectiva, desde la exploración de datos hasta el ajuste final de hiperparámetros. Está organizado en el orden en que se deberían aplicar los pasos en un laboratorio real. Los bloques de código son plantillas reutilizables (basadas en PyTorch) que se pueden adaptar al dataset y arquitectura concretos de cada práctica.

---

## 1. Exploración de datos (antes de tocar el modelo)

Antes de entrenar cualquier modelo hay que responder:

- Tipo de problema: regresión, clasificación binaria/multiclase, etc.
- Número de clases o valores a predecir.
- Distribución de clases (¿dataset balanceado o desbalanceado?).
- Si son imágenes: número de canales, resolución, tipo de dato (`uint8`, `float16`...), estadísticos (media, std, min, max).

Si el dataset está desbalanceado, el modelo tenderá a predecir las clases mayoritarias (*bias*). Se puede corregir sobre-muestreando las clases minoritarias, idealmente combinado con *data augmentation*.

```python
import numpy as np

max_value = train_images.max(axis=(0, 1, 2))
min_value = train_images.min(axis=(0, 1, 2))

mean = (train_images / 255).mean(axis=(0, 1, 2))
std = (train_images / 255).std(axis=(0, 1, 2))

unique, counts = np.unique(train_labels, return_counts=True)
```

---

## 2. Validar la red antes de entrenar en serio

Estos pasos detectan la mayoría de errores de implementación antes de perder tiempo con entrenamientos largos.

### 2.1 Comprobar dimensiones

Pasar un tensor de entrada con las dimensiones esperadas (incluyendo el *batch*) y verificar que la salida tiene la forma correcta.

```python
model = build_model()
test_input = torch.randn((64, D_in)).cuda()
test_output = model(test_input)
test_output.shape  # debe coincidir con (batch, n_clases)
```

### 2.2 Fit de una sola muestra

El modelo debe ser capaz de memorizar una única muestra. Si no lo consigue, algo está mal (función de pérdida incorrecta, dimensiones que no cuadran, etc.).

### 2.3 Fit de un solo batch

De la misma manera, el modelo debe poder memorizar un batch completo (por ejemplo, 64 muestras) en pocas épocas con un *learning rate* algo agresivo (p. ej. `lr=0.01` con Adam).

Solo cuando estos dos checks pasan tiene sentido pasar al entrenamiento con un subconjunto y luego con el dataset completo.

---

## 3. Iterar rápido con un subconjunto representativo

Entrenar en el dataset completo para cada combinación de hiperparámetros es caro. Conviene:

1. Extraer un subconjunto pequeño y representativo (p. ej. 5000 muestras de 40000).
2. Probar arquitecturas, *learning rates*, optimizadores, etc. en ese subconjunto.
3. Una vez encontrada una buena combinación, repetir el experimento final con todos los datos.

Las conclusiones obtenidas en el subconjunto (por ejemplo, cuál es el mejor `lr`) normalmente se transfieren bien al dataset completo, y el coste de cómputo es mucho menor.

---

## 4. Función de entrenamiento reutilizable

Esta es la versión más completa, resultado de combinar las tres plantillas originales: incluye entrenamiento/validación por época, guardado del mejor modelo (checkpointing), *early stopping* opcional, *scheduler* opcional y control de verbosidad. El *weight decay* (regularización L2) se aplica directamente en el optimizador, no en esta función.

```python
import torch
import numpy as np
from sklearn.metrics import accuracy_score

def softmax(x):
    return torch.exp(x) / torch.exp(x).sum(axis=-1, keepdims=True)

def fit(model, dataloader, optimizer, scheduler=None, epochs=100,
        log_each=10, early_stopping=0, verbose=1):
    criterion = torch.nn.CrossEntropyLoss()
    l, acc, lr = [], [], []
    val_l, val_acc = [], []
    best_acc, step = 0, 0

    for e in range(1, epochs + 1):
        _l, _acc = [], []
        for param_group in optimizer.param_groups:
            lr.append(param_group['lr'])

        # entrenamiento
        model.train()
        for x_b, y_b in dataloader['train']:
            y_pred = model(x_b)
            loss = criterion(y_pred, y_b)
            _l.append(loss.item())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            y_probas = torch.argmax(softmax(y_pred), axis=1)
            _acc.append(accuracy_score(y_b.cpu().numpy(), y_probas.cpu().detach().numpy()))
        l.append(np.mean(_l))
        acc.append(np.mean(_acc))

        # validación
        model.eval()
        _l, _acc = [], []
        with torch.no_grad():
            for x_b, y_b in dataloader['val']:
                y_pred = model(x_b)
                loss = criterion(y_pred, y_b)
                _l.append(loss.item())
                y_probas = torch.argmax(softmax(y_pred), axis=1)
                _acc.append(accuracy_score(y_b.cpu().numpy(), y_probas.cpu().numpy()))
        val_l.append(np.mean(_l))
        val_acc.append(np.mean(_acc))

        # guardar el mejor modelo visto hasta ahora
        if val_acc[-1] > best_acc:
            best_acc = val_acc[-1]
            torch.save(model.state_dict(), 'ckpt.pt')
            step = 0
            if verbose == 2:
                print(f"Mejor modelo guardado con acc {best_acc:.5f} en epoch {e}")
        step += 1

        if scheduler:
            scheduler.step()

        # early stopping: cortar si no mejora en N epochs seguidas
        if early_stopping and step > early_stopping:
            print(f"Entrenamiento detenido en epoch {e} por no mejorar en {early_stopping} epochs seguidas")
            break

        if not e % log_each and verbose:
            print(f"Epoch {e}/{epochs} loss {l[-1]:.5f} acc {acc[-1]:.5f} "
                  f"val_loss {val_l[-1]:.5f} val_acc {val_acc[-1]:.5f} lr {lr[-1]:.5f}")

    # al terminar, cargar los mejores pesos (no los últimos)
    model.load_state_dict(torch.load('ckpt.pt'))
    return {'epoch': list(range(1, len(l) + 1)), 'loss': l, 'acc': acc,
            'val_loss': val_l, 'val_acc': val_acc, 'lr': lr}
```

**Por qué esta versión y no las otras:** guarda siempre el mejor checkpoint (no solo el último), soporta *scheduler* y *early stopping* de forma opcional (no obligan a usarlos), y permite tres niveles de verbosidad para no saturar la salida durante *hyperparameter search*.

---

## 5. Optimizadores

Todos disponibles en `torch.optim`. De más simple a más sofisticado:

| Optimizador | Idea clave | Cuándo usarlo |
|---|---|---|
| `SGD` | Descenso de gradiente puro | Baseline, o cuando se dispone de mucho tiempo de cómputo (puede igualar o superar a Adam a largo plazo) |
| `SGD` + `momentum` | Acumula "impulso" en direcciones consistentes | Mejora clara sobre SGD puro, casi siempre recomendable si se usa SGD |
| `RMSprop` | Escala el gradiente según la varianza reciente por dimensión | Alternativa a Adam, menos usada en la práctica |
| `Adam` | Combina momentum + RMSprop | **Opción por defecto recomendada** para empezar cualquier problema nuevo |

Las variantes basadas en momentum (Adam, RMSprop, SGD+momentum) convergen mucho más rápido que SGD puro en las primeras épocas.

```python
# recomendado para empezar
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# alternativa clásica, requiere ajustar más el lr
optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
```

---

## 6. Learning rate scheduling

Variar el `lr` durante el entrenamiento en vez de mantenerlo fijo: valores altos al principio (para avanzar rápido) y valores bajos al final (para afinar cerca del óptimo).

```python
# reduce el lr multiplicándolo por 0.1 cada 10-20 epochs
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.1)

# sube y baja el lr cíclicamente
scheduler = torch.optim.lr_scheduler.CyclicLR(
    optimizer, base_lr=0.0001, max_lr=0.01, step_size_up=5, step_size_down=25
)
```

El scheduler se pasa a la función `fit` y se llama con `scheduler.step()` una vez por época (ver plantilla de la sección 4). Es especialmente útil combinado con *transfer learning*.

---

## 7. Normalización

- **Normalizar las entradas** (dividir entre 255, o restar media y dividir entre desviación estándar) es una buena práctica, aunque en imágenes su efecto suele ser modesto.
- **Batch Normalization** (`torch.nn.BatchNorm1d` / `BatchNorm2d`) tiene un efecto mucho más notable: acelera drásticamente la convergencia, aunque también hace que el modelo caiga en *overfitting* antes, por lo que conviene combinarla siempre con alguna técnica de regularización.

```python
def build_model(D_in, H, D_out):
    return torch.nn.Sequential(
        torch.nn.Linear(D_in, H),
        torch.nn.BatchNorm1d(H),
        torch.nn.ReLU(),
        torch.nn.Linear(H, H),
        torch.nn.BatchNorm1d(H),
        torch.nn.ReLU(),
        torch.nn.Linear(H, D_out)
    ).cuda()
```

---

## 8. Regularización (reducir overfitting)

La señal de *overfitting*: la pérdida/acc de entrenamiento mejora continuamente mientras que la de validación se estanca o empeora.

### 8.1 Regularización L2 (weight decay)

Penaliza pesos grandes añadiendo un término a la función de pérdida. En PyTorch se activa directamente en el optimizador:

```python
optimizer = torch.optim.SGD(model.parameters(), lr=0.1, weight_decay=0.01)
```

Rango típico: `0.001` - `0.01`. Se usa sobre todo con SGD (con Adam es menos habitual, aunque existe `AdamW` para ese caso).

### 8.2 Early stopping

Guardar el checkpoint cada vez que mejora la métrica de validación y, al terminar, quedarse con esos pesos (no con los últimos). Opcionalmente, cortar el entrenamiento si no hay mejora tras N epochs seguidas. Ya está integrado en la función `fit` de la sección 4 mediante el parámetro `early_stopping`. **Es una técnica casi obligatoria, siempre recomendable usarla.**

### 8.3 Dropout

Apaga aleatoriamente neuronas durante el entrenamiento, forzando al modelo a no depender de rutas específicas. Sobre todo útil en arquitecturas `MLP`; menos relevante en redes convolucionales.

```python
def build_model(D_in, H, D_out, p=0.5):
    return torch.nn.Sequential(
        torch.nn.Linear(D_in, H),
        torch.nn.ReLU(),
        torch.nn.Dropout(p),
        torch.nn.Linear(H, H),
        torch.nn.ReLU(),
        torch.nn.Dropout(p),
        torch.nn.Linear(H, D_out)
    ).cuda()
```

Importante: `model.train()` activa el Dropout, `model.eval()` lo desactiva. Olvidar este cambio de modo es una fuente común de errores.

### 8.4 Más datos

La forma más efectiva de reducir overfitting, y la única que además mejora las métricas por sí sola (las demás técnicas reducen el overfitting pero no garantizan mejor rendimiento). No siempre es factible conseguir más datos.

### 8.5 Data augmentation

Aplicar transformaciones aleatorias a cada muestra (recortes, flips, cambios de color/brillo) para que el modelo nunca vea exactamente la misma imagen dos veces. Muy recomendable siempre que se trabaje con imágenes. Requiere entrenamientos más largos para aprovechar todo su potencial. Librería recomendada: `albumentations`.

```python
from albumentations import Compose, RandomCrop, Resize, HorizontalFlip, ToGray, RGBShift, OneOf

trans = Compose([
    RandomCrop(24, 24),
    Resize(32, 32),
    HorizontalFlip(),
    OneOf([
        ToGray(p=0.2),
        RGBShift(p=0.3)
    ])
])

class Dataset(torch.utils.data.Dataset):
    def __init__(self, X, Y, trans=None):
        self.X = X
        self.Y = Y
        self.trans = trans
    def __len__(self):
        return len(self.X)
    def __getitem__(self, ix):
        img = self.X[ix]
        if self.trans:
            img = self.trans(image=img)["image"]
        img = torch.from_numpy(img / 255.).float().cuda().view(-1)
        label = torch.tensor(self.Y[ix]).long().cuda()
        return img, label
```

### Resumen de prioridad de uso

- Uso prácticamente obligado: **early stopping** y, en imágenes, **data augmentation**.
- Uso condicional según el caso, pero siempre vale la pena probarlas: **L2 / weight decay**, **Dropout**.
- Reducir overfitting no implica automáticamente mejores métricas; solo indica que el modelo generaliza mejor.

---

## 9. Tuneado de hiperparámetros

Dos estrategias:

- **Grid search**: probar todas las combinaciones posibles de un conjunto predefinido de valores. Simple pero puede ser muy costoso.
- **Random search** (recomendado): definir un espacio de búsqueda y probar N combinaciones aleatorias dentro de ese espacio. Más eficiente que grid search para el mismo presupuesto de cómputo.

```python
import random

bss = [16, 32, 64, 128, 256]
lrs = [0.01, 0.005, 0.001, 0.0003, 0.0001]
n = 5
resultados = []

for i in range(n):
    lr = random.choice(lrs)
    bs = random.choice(bss)
    dataloader = {
        'train': torch.utils.data.DataLoader(dataset['train'], batch_size=bs, shuffle=True),
        'val': torch.utils.data.DataLoader(dataset['val'], batch_size=1000, shuffle=False)
    }
    model = build_model()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    hist = fit(model, dataloader, optimizer, epochs=20, verbose=0)
    resultados.append({'hist': hist, 'lr': lr, 'bs': bs})
```

Hacer esta búsqueda sobre el subconjunto pequeño de datos (sección 3), no sobre el dataset completo.

---

## 10. Configuraciones recomendadas por defecto

Valores de partida razonables cuando no se sabe por dónde empezar:

| Hiperparámetro | Valor recomendado |
|---|---|
| Optimizador | `Adam`, `lr=0.001` |
| Alternativa | `SGD` + `momentum=0.9`, con scheduler |
| Batch size | 16, 32 o 64 (tender al mayor que quepa en memoria de la GPU, ajustando el `lr`) |
| Weight decay (si se usa SGD) | 0.001 - 0.01 |
| Batch Norm | Sí, casi siempre útil |
| Dropout (en MLP) | `p=0.5` como punto de partida |
| Early stopping | patience de 10-20 epochs |
| Data augmentation | Sí, si se trabaja con imágenes |

---

## 11. Transfer learning (mención)

La técnica más potente para acelerar el entrenamiento y necesitar menos datos: partir de una red ya entrenada en otro dataset (idealmente similar al propio) en lugar de inicializar los pesos desde cero. Combina muy bien con *learning rate scheduling*. Se trata en detalle en labs específicos de transfer learning.

---

## 12. Checklist / receta resumida para un laboratorio nuevo

1. Explorar los datos: tipo de problema, clases, balance, estadísticos.
2. Construir el modelo y comprobar las dimensiones de entrada/salida con un tensor de prueba.
3. Verificar que el modelo puede memorizar una sola muestra.
4. Verificar que el modelo puede memorizar un solo batch.
5. Extraer un subconjunto representativo de datos para iterar rápido.
6. Elegir un optimizador de partida (`Adam, lr=0.001`) y entrenar unas pocas epochs para validar que todo funciona.
7. Añadir `Batch Norm` a la arquitectura.
8. Añadir regularización: `data augmentation` (si hay imágenes) + `early stopping` siempre; `Dropout` y/o `weight decay` según haga falta.
9. Hacer *random search* de hiperparámetros (batch size, lr) sobre el subconjunto.
10. Repetir el mejor entrenamiento con el dataset completo.
11. (Opcional) Añadir *learning rate scheduling*.
12. (Opcional) Si existe una red preentrenada aplicable, considerar *transfer learning*.
