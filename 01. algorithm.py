# Решение "в лоб", чтобы понять правильно ли я понимаю задачу
# Решение не оптимально
# Сложность O(n)

def multiplicate(A: list[int]):
    L = []
    X = 1
    for i in range(len(A)):
        X *= A[i]
    for j in range(len(A)):
        L.append(int(X/A[j]))
    return print(L)

lst = [1,2,3,4]
multiplicate(lst)


# более оптимизированное решение
# сложность O(n)
def multiplicate_2(A):
    n = len(A)
    
    # Массив для результата
    result = [1] * n
    
    # Произведение всех чисел слева от текущего элемента
    left_prod = 1
    for i in range(n):
        result[i] = left_prod
        left_prod *= A[i]
    
    # Произведение всех чисел справа от текущего элемента
    right_prod = 1
    for i in range(n-1, -1, -1):
        result[i] *= right_prod
        right_prod *= A[i]
    
    return result

lst = [1,2,3,4]
multiplicate_2(lst)