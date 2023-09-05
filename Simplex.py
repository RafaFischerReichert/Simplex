import numpy as np
import sys


def inversa(matriz):
    # Checando se é quadrada
    if matriz.shape[0] != matriz.shape[1]:
        raise ValueError("Matriz inserida não é quadrada.")

    # Checando determinante
    if np.linalg.det(matriz) == 0:
        raise ValueError("Determinante da matriz é 0.")

    # Obtendo grau
    grau = matriz.shape[0]

    # Adicionando matriz identidade do mesmo grau
    aumentada = np.hstack((matriz, np.identity(grau)))

    # Eliminação de Gauss
    for i in range(grau):
        # Pivoteando linha: encontrando pivô
        pivo_linha = i
        for j in range(i + 1, grau):
            if abs(aumentada[j, i]) > abs(aumentada[pivo_linha, i]):
                pivo_linha = j

        # Trocando linha pelo pivô
        aumentada[[i, pivo_linha]] = aumentada[[pivo_linha, i]]

        # Escalando pivô
        pivot = aumentada[i, i]
        aumentada[i] /= pivot

        # Realizando a eliminação
        for j in range(grau):
            if j != i:
                fator = aumentada[j, i]
                aumentada[j] -= fator * aumentada[i]

    mat_inversa = aumentada[:, grau:]
    return mat_inversa


# Objetivo: min(2x1 + 3x2)

# Restrições:
# 2x1 + 3x2 >= 5
# x1 - x2 <= 4
# 3x1 - 2x2 = 6
# x1, x2 >= 0

tipo_problema = "min"  # Pode ser max ou min

coeff_objetivo = np.array([3, 2, 0, 0, 0])  # Coeficientes da função z

coeff_restricoes = np.array(
    [[2, 3], [1, -1], [3, -2]]
)  # Coeficientes das restrições

b = np.array([-5, 4, 6])  # Termo independente das restrições -- SINAIS > E >= EXIJEM QUE b*=-1


num_restricoes = coeff_restricoes.shape[0]
num_variaveis = coeff_restricoes.shape[1]
num_basicos = np.count_nonzero(coeff_objetivo)
basicos = np.nonzero(coeff_objetivo)[0]  # Converte para array de índices
nao_basicos = np.where(coeff_objetivo == 0)[
    0
]  # Converte para array de índices
diff = num_restricoes - num_basicos

if num_basicos == 0:
    print("Não há valores básicos!")
    sys.exit()

if diff < 0:
    print("Há mais valores básicos do que restrições!")
    sys.exit()

# Verificação da necessidade da Fase I
if tipo_problema == "max":
    coeff_objetivo *= (
        -1
    )  # Multiplicar a função objetivo por -1 para transformar em minimização

if (b < 0).any():
    # Multiplicar restrições por -1 e inverter a direção das desigualdades
    coeff_restricoes *= -1
    b *= -1

if (coeff_restricoes > 0).any() or (coeff_restricoes == 0).any():
    # Há pelo menos uma restrição com sinal >, >= ou =, vá para a Fase I
    fase1 = True
else:
    fase1 = False

if fase1:
    # FASE I: Formulação do problema artificial
    m = num_restricoes  # Número de restrições originais
    n = (
        num_variaveis + m
    )  # Número total de variáveis (variáveis originais + variáveis artificiais)

    # Atualize as dimensões e os vetores
    dim_B = m
    dim_N = n - m

    # Construa a matriz B e a matriz N
    matriz_B = np.identity(m)
    matriz_N = np.hstack((coeff_restricoes, np.identity(m)))

    # Inicialize os vetores básicos e não-básicos
    basicos = np.arange(m)
    nao_basicos = np.arange(m, n)

    # Inicialize cB e cN
    cB = np.zeros(m)
    cN = coeff_objetivo

    cN_predict = np.zeros(dim_N)

    aN = np.transpose(matriz_N)

    iteration = 1

    while True:
        # Passo 1
        x_basicos_pred = np.dot(inversa(matriz_B), b)
        x_nao_basicos_pred = np.zeros(dim_N)

        # Passo 2
        # 2.1
        cB_transposta = np.transpose(cB)
        lambda_transposta = np.dot(cB_transposta, inversa(matriz_B))
        lambda_vetor = np.transpose(lambda_transposta)

        # 2.2
        for i in range(dim_N):
            cN_predict[i] = cN[i] - np.dot(lambda_transposta, aN[i])

        # 2.3
        cNk = np.min(cN_predict)
        k = np.argmin(cN_predict)

        # Passo 3
        if cNk >= 0:
            if (basicos >= num_variaveis).any():
                print("Problema infactível - Ainda há variáveis artificiais na base.")
                sys.exit()
            else:
                # Se não houver variáveis artificiais na base, vá para a Fase II
                break

        # Passo 4
        y = inversa(matriz_B) * aN[k]

        # Passo 5
        if (y <= 0).any():
            print(
                "Problema não tem solução ótima finita -- Problema Original Infactível."
            )
            sys.exit()
        else:
            epsilon = np.min(np.divide(x_basicos_pred, y))
            l = np.argmin(np.divide(x_basicos_pred, y))

        # Passo 6
        coluna_temporaria = matriz_B[:, l].copy()
        matriz_B[:, l] = matriz_N[:, k]
        matriz_N[:, k] = coluna_temporaria

        # Atualizar basicos e nao_basicos
        basicos[basicos == l] = k
        nao_basicos[nao_basicos == k] = l

        iteration += 1

    # Se a Fase I terminar e ainda houver variáveis artificiais na base, o problema é infactível
    if (basicos >= num_variaveis).any():
        print("Problema infactível.")
        sys.exit()

    # Se não houver mais variáveis artificiais na base, vá para a Fase II


# FASE 2
# Construa as matrizes B e N para a Fase II
matriz_B = coeff_restricoes[:, basicos]
matriz_N = coeff_restricoes[:, nao_basicos - num_variaveis]

dim_B = matriz_B.shape[1]
dim_N = matriz_N.shape[1]

cB = np.split(coeff_objetivo, [dim_B])[0]
cN = np.split(coeff_objetivo, [dim_B])[1]
cN_predict = np.zeros(dim_N)

aN = np.transpose(matriz_N)

while True:
    # Passo 1
    x_basicos_pred = np.dot(inversa(matriz_B), b)
    x_nao_basicos_pred = np.zeros(dim_N)

    # Passo 2
    # 2.1
    cB_transposta = np.transpose(cB)
    lambda_transposta = np.dot(cB_transposta, inversa(matriz_B))
    lambda_vetor = np.transpose(lambda_transposta)

    # 2.2
    for i in range(dim_N):
        cN_predict[i] = cN[i] - np.dot(lambda_transposta, aN[i])

    # 2.3
    cNk = np.min(cN_predict)
    k = np.argmin(cN_predict)

    # Passo 3
    if cNk >= 0:
        print(f"Solução ótima encontrada na iteração {iteration}.")
        break

    # Passo 4
    y = inversa(matriz_B) * aN[k]

    # Passo 5
    if (y <= 0).any():
        print("Problema não tem solução ótima finita -- f(x) tende a -infinito.")
        sys.exit()
    else:
        epsilon = np.min(np.divide(x_basicos_pred, y))
        l = np.argmin(np.divide(x_basicos_pred, y))

    # Passo 6
    coluna_temporaria = matriz_B[:, l].copy()
    matriz_B[:, l] = matriz_N[:, k]
    matriz_N[:, k] = coluna_temporaria

    # Atualizar basicos e nao_basicos
    basicos[basicos == l] = k
    nao_basicos[nao_basicos == k] = l

    iteration += 1

# Após a conclusão da Fase II, você pode calcular a solução ótima
resultado = np.concatenate((x_basicos_pred, x_nao_basicos_pred))

resultado = np.transpose(resultado)

solucao_final = np.dot(coeff_objetivo, resultado)

print("A solução final é", solucao_final, "gerada pelos seguintes valores:", resultado)
