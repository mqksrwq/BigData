# Список из 10 вещественных чисел
numbers = [23.5, 67.2, 12.8, 89.1, 45.3, 76.9,
           34.7, 91.4, 28.6, 62.3]

# Второй список
filtered_numbers = [num for num in numbers if num > 50]

print("Элементы больше 50:", filtered_numbers)

total_sum = sum(filtered_numbers)
print(f"Сумма элементов: {total_sum}")
