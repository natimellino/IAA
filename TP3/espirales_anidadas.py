import random
import math
import matplotlib.pyplot as plt
import pandas as pd

# Las funciones que representan nuestras curvas
def curve1(theta):
    return theta / (4 * math.pi)

def curve2(theta):
    return (theta + math.pi) / (4 * math.pi)

def generate_points_c(n):

    len0 = n // 2
    len1 = n - len0

    class0 = 0
    class1 = 0

    # Radio y centro del circulo
    radius = 1
    x, y = (0, 0)

    # Guardamos las coordenadas x e y respectivamente
    xs = []
    ys = []

    clases = []
    # Tamaño random de puntos a generar
    # for i in range(0, n):
    while (class0 < len0) or (class1 < len1):
    # random.random() devuelve numeros entre 0.0 y 1.0
        theta = 2 * math.pi * random.random()
        r = radius * math.sqrt(random.random())

        # clase = 0
        # Sumando de a 0.5 vamos dando vuelta alrededor de todo el espiral (esto 
        # se puede ver analizando una gráfica de la curva1) y vemos 
        # si el radio cae en alguna de las franjas correctas, y de ser así
        # le asignamos la clase 1 a ese punto, en caso contrario esta permanecerá
        # en 0.
        flag = False
        for theta0 in [theta + (2 * math.pi * i) for i in range(-1, 6)]:
            if (curve1(theta0) < r and r < curve2(theta0)):
                if class1 < len1:
                    class1 += 1
                    # Convertimos las coordenadas polares a cartesianas y almacenamos.
                    xs.append(r * math.cos(theta) + x)
                    ys.append(r * math.sin(theta) + y)
                    # Guadamos la clase correspondiente al punto generado.
                    clases.append(1)
                flag = True
                continue
        if class0 < len0 and not flag:
            class0 += 1
            # Convertimos las coordenadas polares a cartesianas y almacenamos.
            xs.append(r * math.cos(theta) + x)
            ys.append(r * math.sin(theta) + y)
            # Guadamos la clase correspondiente al punto generado.
            clases.append(0)

    # Generamos el dataframe
    points = {'x': xs, 'y': ys, 'Class': clases}
    df = pd.DataFrame(points)
    return df
