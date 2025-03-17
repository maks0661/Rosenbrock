ГА для решения задач оптимизации. 

Генетические алгоритмы(ГА) - методы вычислений, используют принципы естественного отбора для поиска оптимальныз значений. 
Для примера я рассматриваю задачу нахождения максимума ф-ии. Розенброка f(x,y)=(1-x)^2 + 100(y-(x^2))^2

У этой ф-ии. есть глобальный минимум при х=1 и у=1. из-за сложной формы найти его может быть сложно для классических методов оптимизации.

-----------------------------------------------------------------------------

Алгоритм нахожденния:

·Ф-я Розенброка: определяет ф-ю, которую хотим оптимизировать.

·Создание начальной популяции: создает случайную популяцию индивидов (точек в пространстве).

·Оценка приспособленности: вычисляет значение ф-ии Розенброка для каждого индивида.

·Селекция: Выбирает лучших индивидов на основе их приспособленности.

·Кроссовер: Создает новое потомство путем скрещивания родителей.

·Мутация: Вносит случайные изменения в потомство с определенной вероятностью.

·Эволюция: Проводит несколько поколений, улучшая популяцию.

·График: Отображает прогресс фитнес-функции на каждом поколении.

-----------------------------------------------------------------------------

·Популяция - набор возможных решений задачи.

Каждый индивид = потенциальное решение задачи, может быть числовыми значениями, битами и тп.

·если рассматривать задачу оптимизации ф-ии. Розенброка, то каждый индивид - пара значений х и у, которые представляют точку в двумерном пространстве.
·популяция - коллекция таких точек.

Идея в том что начальная популяция создается случайным образом, а затем алгоритм эвлюционирует.

-----------------------------------------------------------------------------

·Индивид - одно возможное решение, представленное в форме ГС(генетической структуры).

·Генетическая структура(ГС) - набор параметров(генов).

Например, в задаче с ф-ей. Розенброка:

- индивид = (x,y) - х,у - это гены индвида.

Каждый индвид оценивается с помощью фитнес-функции, которая  определяет, насколько хорошо это решение подходит для задачи.

-----------------------------------------------------------------------------

·Фитнес-функция (функция приспособленности) - метрика для оценки того, насколько хорошо индивидуум популяции решает задачу оптимизации.

Фитнес-функция принимает на вход индивид и возвращает числовое значение.


Если просто, то фитнесс-функция определяет "здоровье" каждого индивида. Чем выше значение фитнес-функции, тем лучше индивид подходит для решения задачи.

-----------------------------------------------------------------------------

Т.к есть "фитнес-функция", то и соответственно есть "фитнес-значения".

-----------------------------------------------------------------------------

·Фитнес-значения - конкретное числовое значение, которое возвращает фитнес-функция для заданного индивида.

Данное значение используется для сравнения индвидов и выбора лучших из них для создания следующего поколения. 

'''
Например, если есть два индвида: 
- индивид а: (х1, у1)
- индвид б: (х2, у2)
  
Можно вычислить их фитнес-значения: 
- фитнес-значение а: fitness(x1, y1)
- фитнес-значение б: fitness(x2, y2)
Если фитнес-значение "а" больше, чем фитнес-значение "б", значит индвид "а" лучше решает задачу и имеет больше шансов пепередать свои "гены"(то бишь параметры) следующему поколению.
'''

-----------------------------------------------------------------------------

·Потомство - новое поколение индвидов, созданное на основе текущей популяции.

Формируется это все путем скрещивания(кроссовера) и мутации родителей из текущей популяции.

'''
Позволяет передавать лучшие характеристики родителей потомству.
Процесс создания потомства:
- скрещивание(кроссовер)
- мутация
 '''

-----------------------------------------------------------------------------

Далее понятие "скрещивание (кроссовер)".
Скрещивание(кроссовер) - создание новых индивидов путем обмена частями генетической информации(ГИ) между родителями.
Например, есть два родителя:
- родитель-1: (x1,y1)
- родитель-2: (x2,y2)
Можно создать потомка, взяв часть генетической информации(ГИ) от каждого родителя.
Например:
- выбираеми случайную точку разделения(например, после первого гена).
- первую часть берем от первого родителя, вторую-от второго.
Результат:
- потомок: (x1,y2)
Что такое гены(параметры, описание ГС) описано выше ↑.

