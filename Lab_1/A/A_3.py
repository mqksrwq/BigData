# Словарь из 6 записей
mydict = {
    '2101BN': 'Чупров И.П.',
    '2102BN': 'Страхов К.И.',
    '2103BN': 'Иванова А.С.',
    '2104BN': 'Петров В.Г.',
    '2105BN': 'Сидорова Е.М.',
    '2106BN': 'Козлов Д.А.'
}

print("Исходный словарь:")
print(mydict)

# Удаление последней записи
mydict.popitem()
print("\nПосле удаления последней записи:")
print(mydict)

# Список номеров заказов
order_numbers = list(mydict.keys())
print("\nСписок номеров заказов:")
print(order_numbers)

# Копия словаря
copy_dict = mydict.copy()
mydict.clear()
print("\nПосле копирования и очистки оригинала:")
print("Копия:", copy_dict)
print("Оригинал:", mydict)

# Получение фамилии
customer = copy_dict.get('2101BN', 'Заказчик не найден')
print("\nЗаказчик по номеру '2101BN':", customer)
