# %%

import pandas as pd
import missingno as msno
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.ticker import (MultipleLocator, FormatStrFormatter, AutoMinorLocator)

pd.set_option('display.max_columns', None)

# %%

all_horses_df = pd.read_csv(r'C:\users\ivnat\projects\netology\pyda-10\statistics_python_hw_1\horse_data.csv', names=[
    'surgery', 'Age', 'Hospital_Number', 'rectal_temperature',
    'pulse', 'respiratory_rate', 'temperature_of_extremities',
    'peripheral_pulse', 'mucous_membranes', 'capillary_refill_time',
    'pain', 'peristalsis', 'abdominal_distension', 'nasogastric_tube',
    'nasogastric_reflux', 'nasogastric_reflux_PH', 'rectal_examination_feces',
    'abdomen', 'packed_cell_volume', 'total_protein', 'abdominocentesis_appearance',
    'abdomcentesis_total_protein', 'outcome', 'surgical_lesion', 'type_of_lesion_1',
    'type_of_lesion_2', 'type_of_lesion_3', 'cp_data'])

# %% md

### Описание

# %%

# Заменяем пропуски значений ? на NaN

all_horses_df = all_horses_df.replace({'?': np.nan})

# %%

all_horses_df.info()

# %% md

## Описание и преобразование колонок

# %%

# Значение некоторых столбцов необходимо из строкового типа преобразовать в float

# %%

all_horses_df[[
    'surgery', 'Age', 'rectal_temperature',
    'pulse', 'respiratory_rate', 'temperature_of_extremities',
    'peripheral_pulse', 'mucous_membranes', 'capillary_refill_time',
    'pain', 'peristalsis', 'abdominal_distension', 'nasogastric_tube',
    'nasogastric_reflux', 'nasogastric_reflux_PH', 'rectal_examination_feces',
    'abdomen', 'packed_cell_volume', 'total_protein', 'abdominocentesis_appearance',
    'abdomcentesis_total_protein', 'outcome', 'surgical_lesion', 'type_of_lesion_1',
    'type_of_lesion_2', 'type_of_lesion_3', 'cp_data'
]] = all_horses_df[[
    'surgery', 'Age', 'rectal_temperature',
    'pulse', 'respiratory_rate', 'temperature_of_extremities',
    'peripheral_pulse', 'mucous_membranes', 'capillary_refill_time',
    'pain', 'peristalsis', 'abdominal_distension', 'nasogastric_tube',
    'nasogastric_reflux', 'nasogastric_reflux_PH', 'rectal_examination_feces',
    'abdomen', 'packed_cell_volume', 'total_protein', 'abdominocentesis_appearance',
    'abdomcentesis_total_protein', 'outcome', 'surgical_lesion', 'type_of_lesion_1',
    'type_of_lesion_2', 'type_of_lesion_3', 'cp_data']].astype('float')

# %%

type(all_horses_df['rectal_temperature'][0])

# %% md

### surgery

# ** Категорийное значение **
# Проводили  ли операцию для лошади?
# surgery?
# 1 = Yes, it had surgery
# 2 = It was treated without surgery

# %%

# заменяем числовое значение на текстовое для удобства

all_horses_df['surgery'] = all_horses_df['surgery'].replace({1: 'surgery', 2: 'no_surgery'})

# %%

all_horses_df['surgery'].value_counts()

# %%

all_horses_df['surgery'].value_counts().plot(kind='bar', title='Horses_surgery')
plt.show()
# большая часть лошадей проходила операцию

# %%

all_horses_df['surgery'].isna().sum()

# %%

# есть 1 пропущенное значение, можно восстановить значение по колонкам: surgical lesion, type of lesion1,2,3

no_surgery_data = all_horses_df[all_horses_df['surgery'].isna()][['surgery', 'surgical_lesion',
                                                                  'type_of_lesion_1', 'type_of_lesion_2',
                                                                  'type_of_lesion_3']]
no_surgery_data.head()
# %% md

# Исходя из данных можно сделать вывод, что лошадь проходила операцию и данные об этом были случайно утеряны.
# Имеет смысл заменить NaN на surgery

# %%

all_horses_df['surgery'].iloc[132] = 'surgery'

# %%

had_surg = all_horses_df['surgery'][all_horses_df['surgery'] == 'surgery'].count()
no_surg = all_horses_df['surgery'][all_horses_df['surgery'] == 'no_surgery'].count()

print(f'Horses that had surgery: {had_surg}, {round(had_surg / all_horses_df["surgery"].count() * 100, 1)}%')
print(f'Horses that hadn\'t surgery: {no_surg}, {round(no_surg / all_horses_df["surgery"].count() * 100, 1)}%')

# %% md

# ** Вывод по этапу: пропусков в данных нет, можно переходить к изучению и использовать при изучении других данных **

# %% md

### Age

# ** Категорийное значение **
# #
# # Возраст лошади
# #
# # 1 = Adult horse
# #
# # 2 = Young( < 6 months)


# %%

horses_age = all_horses_df['Age'].value_counts()

# %% md

# Скорее всего, значение 9 было указано вместо 2 и означает молодую лошадь.
# Заменим значени на текстовые.

# %%

all_horses_df['Age'] = all_horses_df['Age'].replace({1: 'adult', 9: 'young'})

# %%

all_horses_df['Age'].value_counts().plot(kind='bar', title='horses_age')
plt.show()

# %%

adult = all_horses_df['Age'][all_horses_df['Age'] == 'adult'].count()
young = all_horses_df['Age'][all_horses_df['Age'] == 'young'].count()

print(f'Adult horses (>6 months): {adult}, {round(adult / all_horses_df["Age"].count() * 100, 1)}%')
print(f'Young horses (<6 months): {young}, {round(young / all_horses_df["Age"].count() * 100, 1)}%')

# %% md
# ** Вывод по этапу:
# пропусков в данных нет, можно переходить к изучению и использовать при изучении других данных **


# %% md

## Hospital Number

# ** Категорийное значение **
#
# - numeric id
# - the case number assigned to the horse(may not be unique if the horse is treated > 1 time)

# %%

all_horses_df['Hospital_Number'].nunique()

# %% md

# Проверим сколько раз лошади повторно проходили лечение

# %%

surgeries_n = pd.DataFrame(all_horses_df[['Hospital_Number', 'surgery']] \
                           .groupby(by='Hospital_Number').count().value_counts()).reset_index()

# %% md
# 16 лошадей проходили лечение 2 раза.
# 2 раза - это максимальное количество раз, когда лошадь проходила лечение

# %%
print(
    f'Horses with 1 operation: '
    f'{surgeries_n[0][0]}, {round(surgeries_n[0][0] / all_horses_df["Hospital_Number"].count() * 100, 1)}%'
)
print(
    f'Horses with 2 operations: '
    f'{surgeries_n[0][1]}, {round(surgeries_n[0][1] / all_horses_df["Hospital_Number"].count() * 100, 1)}%'
)

# %% md
# ** Вывод по этапу: пропусков в данных нет, можно переходить к изучению и использовать при изучении других данных **


# %% md
### rectal_temperature
#
# - linear
# - in degrees celsius.
# - An elevated temp may occur due to infection.
# - temperature may be reduced when the animal is in late shock
# - normal temp is 37.8
# - this parameter will usually change as the problem progresses eg.may start out normal,
# then become elevated because of the lesion, passing back through the normal range as the horse goes into shock
#
# ** неприрывное **

# %%

print(
    f'Horses without rectal temperature: '
    f'{all_horses_df["rectal_temperature"].isna().sum()}, '
    f'{round(all_horses_df["rectal_temperature"].isna().sum() / all_horses_df["Hospital_Number"].count() * 100, 1)}%'
)

# %%
fig, ax = plt.subplots(figsize=(16, 10))
ax.hist(all_horses_df['rectal_temperature'], bins=40)
ax.xaxis.set_major_locator(MultipleLocator())
ax.xaxis.set_minor_locator(AutoMinorLocator(5))
ax.yaxis.set_minor_locator(AutoMinorLocator())
plt.xlabel('Темпаратура')
plt.ylabel('Количество')
plt.title('Распределение температуры лошадей')
plt.show()

# %% md
# Распределение похоже на нормальное, провалы в графике могут быть свзязаны с пропусками данных


# %% md
# ** Вывод по этапу.В данных много пропусков, необходимо: **
# Исходя из описания повышенная
# температура может быть связана с инфекцией, послеоперационными осложнениями, пониженная с шоком.
# Скорее всего самым оптимальным будет замена пропусков на средние значния внутри групп


# %% md

### pulse
# - linear
# - the heart rate in beats per minute
# - is a reflection of the heart condition: 30 - 40 is normal for adults
# - rare to have a lower than normal rate although athletic horses may have a rate of 20 - 25
# - animals with painful lesions or suffering from circulatory shock may have an elevated heart rate


# %%

print(
    f'Missing values of horses pulse: {all_horses_df["pulse"].isna().sum()}, '
    f'{round(all_horses_df["pulse"].isna().sum() / all_horses_df["Hospital_Number"].count() * 100, 1)} %'
)

# %% md
# Частота пульса может быть связана со следующими условиями:
# - нормальный пульс 30 - 40 уд
# - пониженный пульс у лошадей участвующих в спорте
# - повышенный пульс у лошадей с circulatory shock

# %%

fig2, ax2 = plt.subplots(figsize=(16, 10))

ax2.hist(all_horses_df['pulse'], bins=40)
# ax2.xaxis.set_major_locator(MultipleLocator())
ax2.xaxis.set_minor_locator(AutoMinorLocator())
ax2.yaxis.set_minor_locator(AutoMinorLocator())
plt.xlabel('Пульс')
plt.ylabel('Количество')
plt.title('Распределение частоты пульса у лошадей')
plt.show()

# %% md
# Распределение не является нормальным, скошено слева.
# Имеются значения превышающие нормальные в 4 - 6 раз.

# %% md
# ** Вывод по этапу: **
# лошадей с нормальным пульсом намного меньше, чем с повышенным,
# повышенный пульс у лошадей может быть связан с circulatory shock
# (Cool и cold температура конечностей, rectal temperature, ниже 37.8,
# pulse, выше 30 - 40 mucous membranes, pale pink).
# Пока сложно сказать какой способ работы с пропусками оптимальный.

# %% md

### respiratory_rate
# respiratory rate
# - linear
# - normal rate is 8 to 10
# - usefulness is doubtful due to the great fluctuations


# %%

print(all_horses_df['respiratory_rate'].isna().sum())
print(all_horses_df['respiratory_rate'].isna().sum() / all_horses_df['Hospital_Number'].count() * 100, '%')

# %%

fig3, ax3 = plt.subplots(figsize=(16, 10))

ax3.hist(all_horses_df['respiratory_rate'], bins=40)
# ax3.xaxis.set_major_locator(MultipleLocator())
ax3.xaxis.set_minor_locator(AutoMinorLocator())
ax3.yaxis.set_minor_locator(AutoMinorLocator())
plt.xlabel('respiratory_rate')
plt.ylabel('Количество')
plt.title('Частота respiratory_rate у лошадей')
plt.show()

# %% md
# Распределение не является нормальным, скошено влево.Есть значения превышающие норму в 8 - 10 раз.
# Абсолютное большинство значений превышают норму

# %% md
# ** Вывод по этапу: **
# Полезность сомнительна в связи с большим разбросом значений.
# ** Возможно, стоит удалить весь столбец **

# %% md

### temperature_of_extremities
# - a subjective indication of peripheral circulation
# - possible values:
# 1 = Normal
# 2 = Warm
# 3 = Cool
# 4 = Cold
# - cool to cold extremities indicate possible shock
# - hot extremities should correlate with an elevated rectal temp.

# %%

# заменяем числовое значение на текстовое для удобства

all_horses_df['temperature_of_extremities'] = all_horses_df['temperature_of_extremities'] \
    .replace({1: 'Normal', 2: 'Warm', 3: 'Cool', 4: 'Cold'})

# %%

all_horses_df['temperature_of_extremities'].value_counts()

# %%

all_horses_df['temperature_of_extremities'].value_counts().plot(kind='bar', title='temperature_of_extremities')
plt.show()
# %%

print(
    f"Horses with missing temperature of extremeties: "
    f"{all_horses_df['temperature_of_extremities'].isna().sum()}, "
    f"{round(all_horses_df['temperature_of_extremities'].isna().sum() / all_horses_df['Hospital_Number'].count() * 100, 1)}%"
)

# %% md
# ** Вывод по этапу: **

# ** Значение показателя warm кореллирует с повышенной ректальной температурой. **

# ** Cool и cold c шоком, можно попробовать восстановить по косвенным признакам: **
# - ** rectal temperature, ниже 37.8 **
# - ** pulse, выше 30 - 40 **
# - ** mucous membranes, pale pink **

# %% md

### peripheral_pulse

# - subjective
# - possible values
# are:
# 1 = normal
# 2 = increased
# 3 = reduced
# 4 = absent
# - normal or increased p.p.are indicative of adequate circulation while reduced or absent indicate poor perfusion


# %%

# заменяем числовое значение на текстовое для удобства

all_horses_df['peripheral_pulse'] = all_horses_df['peripheral_pulse'] \
    .replace({1: 'normal', 2: 'increased', 3: 'reduced', 4: 'absent'})

# %%

all_horses_df['peripheral_pulse'].value_counts()

# %%

all_horses_df['peripheral_pulse'].value_counts().plot(kind='bar', title='peripheral_pulse')
plt.show()

# %%
print(
    f"Horses with missing peripheral pulse: {all_horses_df['peripheral_pulse'].isna().sum()}, "
    f"{round(all_horses_df['peripheral_pulse'].isna().sum() / all_horses_df['Hospital_Number'].count() * 100, 1)}%"
)

# %% md
# ** Вывод по этапу: **

# Может быть связано с температурой конечностей temperature_of_extremities,
# cool - redused, cold - absent, normal - normal, warm - increased.
# Можно попробовать восстановить по этим данным


# %% md

### mucous_membranes
# - a subjective measurement of colour
# - possible values are:
# 1 = normal pink
# 2 = bright pink
# 3 = pale pink
# 4 = pale cyanotic
# 5 = bright red / injected
# 6 = dark cyanotic
#
# - 1 and 2 probably indicate a normal or slightly increased circulation
# - 3 may occur in early shock
# - 4 and 6 are indicative of serious circulatory compromise
# - 5 is more indicative of a septicemia

# %%

# заменяем числовое значение на текстовое для удобства

all_horses_df['mucous_membranes'] = all_horses_df['mucous_membranes'] \
    .replace(
    {1: 'normal_pink', 2: 'bright_pink', 3: 'pale_pink', 4: 'pale_cyanotic', 5: 'bright_red', 6: 'dark_cyanotic'})

# %%

all_horses_df['mucous_membranes'].value_counts()

# %%

all_horses_df['mucous_membranes'].value_counts().plot(kind='bar', title='mucous_membranes')
plt.show()

# %%

print(
    f"Horses with missing mucous membranes color data: {all_horses_df['mucous_membranes'].isna().sum()}, "
    f"{round(all_horses_df['mucous_membranes'].isna().sum() / all_horses_df['Hospital_Number'].count() * 100, 1)}%"
)

# %% md
# ** Вывод по этапу: **
# # можно восстановить часть значений по косвенным признакам:
# # - normal pink и bright pink - нормальная и увеличенная циркуляция
# # - pale pink - шок
# # - pale cyanotic и dark cyanotic - сниженная и отсутстсвующая циркуляция
# # - dark cyanotic - сепсис(нет данных)

# %% md

### capillary_refill_time
# - a clinical judgement.The longer the refill, the poorer the circulation
# - possible values 1 = < 3 seconds 2 = >= 3 seconds


# %%

all_horses_df['capillary_refill_time'].value_counts()

# %% md
# Есть значение - 3, не входящее в перечень допустимых категорий - 1 и 2.
# Возможно, 3 - это значение "3 секунды" и его можно отнести к категории 2.


# %%

cap_ref_3 = all_horses_df[all_horses_df['capillary_refill_time'] == 3]

# %% md
# В обоих случаях mucous_membranes pale_pink и peripheral_pulse reduced,
# что может быть связано с проблемами с циркуляцией, так что есть основание предположить,
# что 3 - это "3 секунды" и отнести эти значения ко второй группе

# %%

# заменяем числовое значение на текстовое для удобства

all_horses_df['capillary_refill_time'] = all_horses_df['capillary_refill_time'] \
    .replace({1: 'normal', 2: 'poor', 3: 'poor'})

# %%
all_horses_df['capillary_refill_time'].value_counts().plot(kind='bar', title='capillary_refill_time')
plt.show()

# %%

print(
    f"Horses without capillary refill time data: {all_horses_df['capillary_refill_time'].isna().sum()}, "
    f"{round(all_horses_df['capillary_refill_time'].isna().sum() / all_horses_df['Hospital_Number'].count() * 100, 1)}%"
)

# %% md
# ** Вывод по этапу: **
# Отсутствует 10 % значений, возможно, можно восстановить значение по значениям mucous_membranes и peripheral_pulse
# - mucous_membranes normal pink, bright pink и peripheral_pulse со значениями normal, increased - 1
# - mucous_membranes pale cyanotic, dark cyanotic и peripheral_pulse со значениями reduced, absent - 2

# %% md

### pain
# pain - a subjective judgement of the horse's pain level - possible values:
# # 1 = alert, no pain
# # 2 = depressed
# # 3 = intermittent mild pain
# # 4 = intermittent severe pain
# # 5 = continuous severe pain
# # - should NOT be treated as a ordered or discrete variable!
# # - In general, the more painful, the more likely it is to require surgery
# # - prior treatment of pain may mask the pain level to some extent

# %%

# заменяем числовое значение на текстовое для удобства

all_horses_df['pain'] = all_horses_df['pain'] \
    .replace({1: 'no_pain', 2: 'depressed', 3: 'intermittent_mild_pain',
              4: 'intermittent_severe_pain', 5: 'continuous_severe_pain'})

# %%

all_horses_df['pain'].value_counts()

# %%
all_horses_df['pain'].value_counts().plot(kind='bar', title='pain')
plt.show()

# %%

print(
    f"Horses with missing pain level data: {all_horses_df['pain'].isna().sum()}, "
    f"{round(all_horses_df['pain'].isna().sum() / all_horses_df['Hospital_Number'].count() * 100, 1)}"
)

# %% md
# ** Вывод по этапу: **
# # в описании не указана взаимосвязь pain с другими данными, но чем хуже симптомы, тем сильнее уровень боли,
# # abdominal_distention moderate, severe связан с наличием боли, чем больше, тем выше уровень боли

# %% md

### peristalsis
# - an indication of the activity in the horse's gut.
# As the gut becomes more distended or the horse becomes more toxic, the activity decreases
# - possible values:
# 1 = hypermotile
# 2 = normal
# 3 = hypomotile
# 4 = absent

# %%

# заменяем числовое значение на текстовое для удобства

all_horses_df['peristalsis'] = all_horses_df['peristalsis'] \
    .replace({1: 'hypermotile', 2: 'normal', 3: 'hypomotile', 4: 'absent'})

# %%

all_horses_df['peristalsis'].value_counts()

# %%

all_horses_df['peristalsis'].value_counts().plot(kind='bar', title='peristalsis')
plt.show()
# %%

print(
    f"Horses with missing persistalsis data: {all_horses_df['peristalsis'].isna().sum()}, "
    f"{round(all_horses_df['peristalsis'].isna().sum() / all_horses_df['Hospital_Number'].count() * 100, 1)}%"
)

# %% md
# ** Вывод по этапу: **
# abdominal_distention moderate, severe может быть связан с persistalsis hypomotile, absent

# %% md

### abdominal_distension

# - An IMPORTANT parameter.
# - possible values
# 1 = none
# 2 = slight
# 3 = moderate
# 4 = severe
# - an animal with abdominal distension is likely to be painful and have reduced gut motility.
# - a horse with severe abdominal distension is likely to require surgery just to relieve the pressure

# %%

# заменяем числовое значение на текстовое для удобства

all_horses_df['abdominal_distension'] = all_horses_df['abdominal_distension'] \
    .replace({1: 'none', 2: 'slight', 3: 'moderate', 4: 'severe'})

# %%

all_horses_df['abdominal_distension'].value_counts()

# %%
all_horses_df['abdominal_distension'].value_counts().plot(kind='bar', title='abdominal_distension')
plt.show()
# %%
print(
    f"Horses with missing abdominal distention data: {all_horses_df['abdominal_distension'].isna().sum()}, "
    f"{round(all_horses_df['abdominal_distension'].isna().sum() / all_horses_df['Hospital_Number'].count() * 100, 1)}%"
)

# %% md

# ** Вывод по этапу:  **
# в данных много пропусков, необходимо:
# - abdominal_distension moderate, severe может быть связан с persistalsis hypomotile, absent
# - Значение связано с уровнем боли, чем выше уровень боли, тем вероятнее будут значение severe,
# и верятнее операционное вмешательство

# %% md

### nasogastric_tube
# - this refers to any gas coming out of the tube
# - possible
# values:
# 1 = none
# 2 = slight
# 3 = significant
# - a large gas cap in the stomach is likely to give the horse discomfort

# %%

# заменяем числовое значение на текстовое для удобства

all_horses_df['nasogastric_tube'] = all_horses_df['nasogastric_tube'] \
    .replace({1: 'none', 2: 'slight', 3: 'significant'})

# %%

all_horses_df['nasogastric_tube'].value_counts()

# %%

all_horses_df['nasogastric_tube'].value_counts().plot(kind='bar', title='nasogastric_tube')
plt.show()
# %%

print(
    f"Missing nasogastric tube data: {all_horses_df['nasogastric_tube'].isna().sum()}, "
    f"{round(all_horses_df['nasogastric_tube'].isna().sum() / all_horses_df['Hospital_Number'].count() * 100, 1)}%"
)

# %% md
# ** Вывод по этапу: **
# Есть вероятность, что nasogastric_tube не вводили, поэтому данные отсутсвуют.
# О наличии газов могут свидетельствовать косвенные признаки, например, abdominal_distension и уровень pain.
# Нужно уточнить взаимозависимость данных

# %% md

### nasogastric_reflux

# - possible values
# 1 = none
# 2 = > 1 liter
# 3 = < 1 liter
# - the greater amount of reflux, the more likelihood that there is some serious obstruction to the fluid passage
# from the rest of the intestine

# %%

# заменяем числовое значение на текстовое для удобства

all_horses_df['nasogastric_reflux'] = all_horses_df['nasogastric_reflux'] \
    .replace({1: 'none', 2: 'more_1_liter', 3: 'less_1_liter'})

# %%

all_horses_df['nasogastric_reflux'].value_counts()

# %%
all_horses_df['nasogastric_reflux'].value_counts().plot(kind='bar', title='nasogastric_reflux')
plt.show()
# %%
print(
    f"Missing nasogastric reflux data: {all_horses_df['nasogastric_reflux'].isna().sum()}, "
    f"{round(all_horses_df['nasogastric_reflux'].isna().sum() / all_horses_df['Hospital_Number'].count() * 100, 1)}%"
)

# %% md
# ** Вывод по этапу: **
# Пропусков примерно столько же, сколько и в nasogastric_tube,
# скорее всего эти показатели связаны напрямую - nasogastric_reflux измеряется с помощью nasogastric_tube.
# Показатель nasogastric_reflux больше 1 литра может быть связан с показателем rectal_examination_feces - absent,
# т.к.последнее значение связано с пониженной проходимостью кишечника

# %% md

### nasogastric_reflux_PH

# - linear
# - scale is from 0 to 14 with 7 being neutral
# - normal values are in the 3 to 4 range

# %%

fig4, ax4 = plt.subplots(figsize=(16, 10))
ax4.hist(all_horses_df['nasogastric_reflux_PH'])
ax4.xaxis.set_major_locator(MultipleLocator())
ax4.xaxis.set_minor_locator(AutoMinorLocator(5))
ax4.yaxis.set_minor_locator(AutoMinorLocator())
plt.xlabel('Темпаратура')
plt.ylabel('Количество')
plt.title('Распределение nasogastric_reflux_PH лошадей')
plt.show()

# %%
print(
    f"Missing nasogastric reflux PH data: {all_horses_df['nasogastric_reflux_PH'].isna().sum()}, "
    f"{round(all_horses_df['nasogastric_reflux_PH'].isna().sum() / all_horses_df['Hospital_Number'].count() * 100, 1)}%"
)

# %% md

# ** Вывод по этапу:
# Отсутствует более 80 % значений, так же в описании нет взаимосвязи с остальными данными,
# поэтому восстановление маловероятно.
# Самым лучшим решением тут будет - удаление этих данных **

# %% md

### rectal_examination_feces

# - possible values
# 1 = normal
# 2 = increased
# 3 = decreased
# 4 = absent
# - absent feces probably indicates an obstruction

# %%

# заменяем числовое значение на текстовое для удобства

all_horses_df['rectal_examination_feces'] = all_horses_df['rectal_examination_feces'] \
    .replace({1: 'normal', 2: 'increased', 3: 'decreased', 4: 'absent'})

# %%

all_horses_df['rectal_examination_feces'].value_counts()

# %%

all_horses_df['rectal_examination_feces'].value_counts().plot(kind='bar', title='rectal_examination_feces')
plt.show()

# %%

print(
    f"Missing rectal examination feces data: {all_horses_df['rectal_examination_feces'].isna().sum()}, "
    f"{round(all_horses_df['rectal_examination_feces'].isna().sum() / all_horses_df['Hospital_Number'].count() * 100, 1)}%"
)

# %% md
# ** Вывод по этапу: **
# # rectal_examination_feces absent может быть связан с persistalsis absent и nasogastric_reflux больше 1 литра

# %% md

### abdomen

# - possible values
# 1 = normal
# 2 = other
# 3 = firm feces in the large intestine
# 4 = distended small intestine
# 5 = distended large intestine
# - 3 is probably an obstruction caused by a mechanical impaction and is normally treated medically
# - 4 and 5 indicate a surgical lesion

# %%

# заменяем числовое значение на текстовое для удобства

all_horses_df['abdomen'] = all_horses_df['abdomen'] \
    .replace({1: 'normal', 2: 'other', 3: 'firm_feces_large_intestine',
              4: 'distended_small_intestine', 5: 'distended_large_intestine'})

# %%

all_horses_df['abdomen'].value_counts()

# %%

all_horses_df['abdomen'].value_counts().plot(kind='bar', title='abdomen')
plt.show()
# %%
print(
    f"Missing abdomen data: {all_horses_df['abdomen'].isna().sum()}, "
    f"{round(all_horses_df['abdomen'].isna().sum() / all_horses_df['Hospital_Number'].count() * 100, 1)}%"
)

# %% md

# ** Вывод по этапу: **
# abdomen firm_feces_large_intestine может быть связан с rectal_examination_feces absent
# и nasogastric_reflux больше 1 литра.
# abdomen distended small intestine и distended
# large intestine с хирургическими осложнениями

# %% md

### packed cell volume
# - linear
# - the  # of red cells by volume in the blood
# - normal range is 30 to 50.
# The level rises as the circulation becomes compromised or as the animal becomes dehydrated.

# %%

fig5, ax5 = plt.subplots(figsize=(16, 10))

ax5.hist(all_horses_df['packed_cell_volume'])
ax5.xaxis.set_major_locator(MultipleLocator())
ax5.xaxis.set_minor_locator(AutoMinorLocator(5))
ax5.yaxis.set_minor_locator(AutoMinorLocator())
plt.xlabel('red cells')
plt.ylabel('Количество')
plt.title('Распределение packed_cell_volume лошадей')
plt.show()
# %%
print(
    f"Missing packed cell volume data: {all_horses_df['packed_cell_volume'].isna().sum()}, "
    f"{round(all_horses_df['packed_cell_volume'].isna().sum() / all_horses_df['Hospital_Number'].count() * 100, 1)}%"
)

# %% md

# ** Вывод по этапу: **
# У большинства лошадей нормальный показатель количества red cell. Нужно узнать причину пропусков.
# Повышенный уровень связан с compromised circulation (mucus membranes pale, rectal_examination_feces absent,
# persistalsis absent и nasogastric_reflux больше 1 литра)


# %% md

### total protein
# - linear
# - normal values lie in the 6 - 7.5(gms / dL) range
# - the higher the value the greater the dehydration

# %%

fig6, ax6 = plt.subplots(figsize=(16, 10))

ax6.hist(all_horses_df['total_protein'], bins=25)
ax6.xaxis.set_major_locator(MultipleLocator(3))
ax6.yaxis.set_minor_locator(AutoMinorLocator())
plt.xlabel('total protein')
plt.ylabel('Количество')
plt.title('Распределение total_protein лошадей')
plt.show()

# %%
print(
    f"Missing total rotein data: {all_horses_df['total_protein'].isna().sum()}, "
    f"{round(all_horses_df['total_protein'].isna().sum() / all_horses_df['Hospital_Number'].count() * 100, 1)}%"
)

# %% md

# ** Вывод по этапу: **
# Повышенный уровень total protein связан с обезвоживанием.
# Что связывает его с повышенным уровнем packed_cell_volume

# %% md

### abdominocentesis appearance
# - a needle is put in the horse's abdomen and fluid is obtained from the abdominal cavity
# - possible values:
# 1 = clear
# 2 = cloudy
# 3 = serosanguinous
# - normal fluid is clear while cloudy or serosanguinous indicates a compromised gut


# %%

# заменяем числовое значение на текстовое для удобства

all_horses_df['abdominocentesis_appearance'] = all_horses_df['abdominocentesis_appearance'] \
    .replace({1: 'clear', 2: 'cloudy', 3: 'serosanguinous'})

# %%

all_horses_df['abdominocentesis_appearance'].value_counts()

# %%

all_horses_df['abdominocentesis_appearance'].value_counts().plot(kind='bar', title='abdominocentesis_appearance')
plt.show()
# %%
print(
    f"Missing abdominocentesis appearance data: {all_horses_df['abdominocentesis_appearance'].isna().sum()}, "
    f"{round(all_horses_df['abdominocentesis_appearance'].isna().sum() / all_horses_df['Hospital_Number'].count() * 100, 1)}%"
)

# %% md

# ** Вывод по этапу: **
# Более половины значений нет.Возможно это связано с тем, что эту процедуру не делали.
# Значения распределены примерно поровну. Если не будет найдена причина пропусков, возможно,
# лучшим выходом будет удаление данных.

# %% md

### abdomcentesis total protein

# - linear
# - the higher the level of protein the more likely it is to have a compromised gut.
# Values are in gms / dL

# %%

fig7, ax7 = plt.subplots(figsize=(16, 10))

ax7.hist(all_horses_df['abdomcentesis_total_protein'], bins=25)
ax7.xaxis.set_major_locator(MultipleLocator())
ax7.xaxis.set_minor_locator(AutoMinorLocator())
ax7.yaxis.set_minor_locator(AutoMinorLocator())
plt.xlabel('abdomcentesis_total_protein')
plt.ylabel('Количество')
plt.title('Распределение abdomcentesis_total_protein лошадей')
plt.show()

# %%
print(
    f"Missing abdomcentesis total protein data: {all_horses_df['abdomcentesis_total_protein'].isna().sum()}, "
    f"{round(all_horses_df['abdomcentesis_total_protein'].isna().sum() / all_horses_df['Hospital_Number'].count() * 100, 1)}%"
)

# %% md

# ** Вывод по этапу: **

# Больше половины пропущено, нет референсных значений.
# Показывает вероятность compromised gut, как и многие другие показатели.
# Возможно лучший выход - удаление данных.

# %% md

### outcome
# - what eventually happened to the horse?
# - possible values:
# 1 = lived
# 2 = died
# 3 = was euthanized

# %%

# заменяем числовое значение на текстовое для удобства

all_horses_df['outcome'] = all_horses_df['outcome'] \
    .replace({1: 'lived', 2: 'died', 3: 'euthanized'})

# %%

all_horses_df['outcome'].value_counts()

# %%

all_horses_df['outcome'].value_counts().plot(kind='bar', title='outcome')
plt.show()

# %%
print(
    f"Missing outcome data: {all_horses_df['outcome'].isna().sum()}, "
    f"{round(all_horses_df['outcome'].isna().sum() / all_horses_df['Hospital_Number'].count() * 100, 1)}%"
)

# %%

# Нет информации только об одной лошади, посмотрим на данные подробнее

no_outcome = all_horses_df[all_horses_df['outcome'].isna()]

# %%
no_outcome_num = all_horses_df[all_horses_df['Hospital_Number'] == 534572]

# %% md

# ** Вывод по этапу: **
# Отсутствует одно значение. Лошадь проходила лечение только 1 раз.
# Есть два варианта - попробовать предсказать исход по имеющимся данным, либо удалить всю строку.

# %% md

### surgical lesion?
# - retrospectively, was the problem(lesion) surgical?
# - all cases are either operated upon or autopsied so that this value and the lesion type are always known
# - possible
# values:
# 1 = Yes
# 2 = No

# %%

# заменяем числовое значение на текстовое для удобства

all_horses_df['surgical_lesion'] = all_horses_df['surgical_lesion'] \
    .replace({1: 'Yes', 2: 'No'})

# %%

all_horses_df['surgical_lesion'].value_counts()

# %%
all_horses_df['surgical_lesion'].value_counts().plot(kind='bar', title='surgical_lesion')
plt.show()

# %%

print(
    f"Missing surgical lesion data: {all_horses_df['surgical_lesion'].isna().sum()}, "
    f"{round(all_horses_df['surgical_lesion'].isna().sum() / all_horses_df['Hospital_Number'].count() * 100, 1)}%"
)
# %% md

# ** Вывод по этапу: **
# данные полные

# %% md

### types of lesion
# - first number is site of lesion
# 1 = gastric
# 2 = sm intestine
# 3 = lg colon
# 4 = lg colon and cecum
# 5 = cecum
# 6 = transverse colon
# 7 = retum / descending colon
# 8 = uterus
# 9 = bladder
# 11 = all intestinal sites
# 00 = none
#
# - second number is type
# 1 = simple
# 2 = strangulation
# 3 = inflammation
# 4 = other
#
# - third number is subtype
# 1 = mechanical
# 2 = paralytic
# 0 = n / a
#
# - fourth number is specific code
# 1 = obturation
# 2 = intrinsic
# 3 = extrinsic
# 4 = adynamic
# 5 = volvulus / torsion
# 6 = intussuption
# 7 = thromboembolic
# 8 = hernia
# 9 = lipoma / slenic
# incarceration
# 10 = displacement
# 0 = n / a

# %%

all_horses_df[(all_horses_df['surgical_lesion'] == 'Yes') & (all_horses_df['type_of_lesion_1'] == 0.0)] \
    ['type_of_lesion_1']

# %%

all_horses_df.iloc[145]

# %% md

# Как минимум одно значение отсутсвует.
# Отсутствие других значений определить наверняка нельзя, т.к.осложнения могут отсутствовать

# %%
all_horses_df[all_horses_df['type_of_lesion_1'] == 0.0]['type_of_lesion_1'].count()
# без осложнений
# %% md

# ** Вывод по этапу: **
# Оставляем данные в том виде в котором они есть,
# т.к. определить наличие/отсутствие осложнений по имеющимся данным нельзя.

# %% md

### Работа с пропусками
# 2 часть:
# - msno
# после определить - mcar, mar, mnar
# процент пропущенных данных
# выбросы
# взаимосвязь с другими данными исходя из описания

# %%
msno.matrix(all_horses_df)
plt.show()

# Есть закономерности в пропусках данных

# %%
msno.bar(all_horses_df)
plt.show()

# %% md

# Удалим строки в которых отсутсвует больше 70 % значений

all_horses_df.shape

# %% md

# В датасете 9 столбцов без пропусков, 19 столбцов с пропусками.
# Удалим строки в которых отсутсвует больше 15 строк (ок 80%), остальные попытаемся восстановить.


# %%

all_horses_df = all_horses_df.dropna(thresh=13)  # Оставляем строки где не пропущено 13 значений и больше

# %%
# Сортируем значения: строки по убыванию пропусков

all_horses_df_sorted = all_horses_df.sort_values(by=[
    'rectal_temperature',
    'pulse', 'respiratory_rate', 'temperature_of_extremities',
    'peripheral_pulse', 'mucous_membranes', 'capillary_refill_time',
    'pain', 'peristalsis', 'abdominal_distension', 'nasogastric_tube',
    'nasogastric_reflux', 'nasogastric_reflux_PH', 'rectal_examination_feces',
    'abdomen', 'packed_cell_volume', 'total_protein', 'abdominocentesis_appearance',
    'abdomcentesis_total_protein', 'outcome', 'surgical_lesion', 'type_of_lesion_1',
    'type_of_lesion_2', 'type_of_lesion_3'], na_position='first')

# %%
# удаляем колонки с большим количеством пропусков и неинформативными данными:
# respiratory_rate, nasogastric_reflux_PH, cp_data
all_horses_df = all_horses_df.drop(columns=['respiratory_rate', 'nasogastric_reflux_PH', 'cp_data'])

# %%
msno.heatmap(all_horses_df)
plt.show()

# Есть корелляция в пропусках данных

# %%
### Рассмотрим каждый столбец в отдельности вместе со связанными данными (из описания и корелляции пропусков)

# %%
### rectal_temperature
# Нормальное значение 37.8
# отсутствие rectal_temperature кореллирует с отсутствием значения pulse
# Повышенная температура кореллирует с инфекцией и осложнениями
# Пониженная с шоком

# посмотрим как взаимосвязаны пропуски
msno.matrix(all_horses_df[['rectal_temperature', 'pulse']])
# Есть взаимосвязь пропусков в данных, но его причина исходя из имеющихся в описаниях значений данных
# и визуализации пропусков не ясна.
# MAR

# Попробуем разделить на группы и определить средние значения внутри групп
# Инфекция и осложнения - Warm temperature_of_extremities, mucous_membranes bright red / injected и dark cyanotic,
# abdomen distended small intestine и distended large intestine

# Шок - пульс выше 40 уд, Cool и cold temperature_of_extremities, mucous membranes pale pink
# %%
norm_rect_temp = all_horses_df[all_horses_df['rectal_temperature'] == 37.8]
low_rect_temp = all_horses_df[all_horses_df['rectal_temperature'] < 37.8]
high_rect_temp = all_horses_df[all_horses_df['rectal_temperature'] > 37.8]

# %%
# Значения должны различаться согласно описанию выше для выбранных групп

fig8, axs8 = plt.subplots(3, 4, figsize=(24, 16))
low_rect_temp['temperature_of_extremities'].value_counts().plot(kind='bar', ax=axs8[0, 0])
low_rect_temp['mucous_membranes'].value_counts().plot(kind='bar', ax=axs8[0, 1])
low_rect_temp['abdomen'].value_counts().plot(kind='bar', ax=axs8[0, 2])
low_rect_temp['pulse'].plot(kind='box', ax=axs8[0, 3])

high_rect_temp['temperature_of_extremities'].value_counts().plot(kind='bar', ax=axs8[1, 0])
high_rect_temp['mucous_membranes'].value_counts().plot(kind='bar', ax=axs8[1, 1])
high_rect_temp['abdomen'].value_counts().plot(kind='bar', ax=axs8[1, 2])
high_rect_temp['pulse'].plot(kind='box', ax=axs8[1, 3])

norm_rect_temp['temperature_of_extremities'].value_counts().plot(kind='bar', ax=axs8[2, 0])
norm_rect_temp['mucous_membranes'].value_counts().plot(kind='bar', ax=axs8[2, 1])
norm_rect_temp['abdomen'].value_counts().plot(kind='bar', ax=axs8[2, 2])
norm_rect_temp['pulse'].plot(kind='box', ax=axs8[2, 3])

axs8[0, 0].set_title("low rectal temperature, temperature_of_extremities")
axs8[0, 1].set_title("low rectal temperature, mucous_membranes")
axs8[0, 2].set_title("low rectal temperature, abdomen")
axs8[0, 3].set_title("low rectal temperature, pulse")

axs8[1, 0].set_title("high rectal temperature,temperature_of_extremities")
axs8[1, 1].set_title("high rectal temperature, mucous_membranes")
axs8[1, 2].set_title("high rectal temperature, abdomen")
axs8[1, 3].set_title("high rectal temperature, pulse")

axs8[2, 0].set_title("norm rectal temperature, temperature_of_extremities")
axs8[2, 1].set_title("norm rectal temperature, mucous_membranes")
axs8[2, 2].set_title("norm rectal temperature, abdomen")
axs8[2, 3].set_title("norm rectal temperature, pulse")

plt.show()

# сравнение данных показывает,
# что вне зависимости от повышенной или пониженной температуры данные (практически) не отличаются,
# поэтому деление на группы невозможно,
# выбираем замену пропусков на средние значения по всем данным
# %%
# Mean или median?
all_horses_df.boxplot(column=['rectal_temperature'])
plt.show()
# Есть несколько выбросов как в одну так и в другую сторону, но они практически симметричны,
# и поэтому не должны давать сильное искажение mean, распределение похоже на нормальное,
# поэтому mean и median не должны сильно различатсья
# %%
print(
    f"Mean rectal temp: {round(all_horses_df['rectal_temperature'].mean(), 3)} \n"
    f"Median rectal temp: {all_horses_df['rectal_temperature'].median()}"
)

# %%

all_horses_df['rectal_temperature'].fillna(
    round(all_horses_df['rectal_temperature'].mean(), 1), inplace=True)

# %%
print(
    f"Mean rectal temp: {round(all_horses_df['rectal_temperature'].mean(), 3)} \n"
    f"Median rectal temp: {all_horses_df['rectal_temperature'].median()}"
)

all_horses_df.boxplot(column=['rectal_temperature'])
plt.show()

# %% md
###Pulse

# %%
# нормальное значение 30-40 уд
# повышенный пульс связан с circulatory shock
# (Cool и cold температура конечностей, rectal temperature, ниже 37.8,
# pulse, выше 30 - 40 mucous membranes, pale pink)

# Пропуски незначительно коррелируют с rectal temperature, но предыдущее изучение данных не показало причины взаимосвязи
# MAR

all_horses_df.boxplot(column=['pulse'])
plt.show()

# %%
# Попробуем разделить на группы и определить средние значения внутри групп

norm_pulse = all_horses_df[all_horses_df['pulse'] <= 40]
high_pulse = all_horses_df[all_horses_df['pulse'] > 40]

# %%

# Значения должны различаться согласно описанию выше для выбранных групп

fig9, axs9 = plt.subplots(2, 3, figsize=(24, 16))
norm_pulse['temperature_of_extremities'].value_counts().plot(kind='bar', ax=axs9[0, 0])
norm_pulse['mucous_membranes'].value_counts().plot(kind='bar', ax=axs9[0, 1])
norm_pulse['rectal_temperature'].plot(kind='box', ax=axs9[0, 2])

high_pulse['temperature_of_extremities'].value_counts().plot(kind='bar', ax=axs9[1, 0])
high_pulse['mucous_membranes'].value_counts().plot(kind='bar', ax=axs9[1, 1])
high_pulse['rectal_temperature'].plot(kind='box', ax=axs9[1, 2])

axs9[0, 0].set_title("normal pulse, temperature_of_extremities")
axs9[0, 1].set_title("normal pulse, mucous_membranes")
axs9[0, 2].set_title("normal pulse, rectal temperature")

axs9[1, 0].set_title("high pulse, temperature_of_extremities")
axs9[1, 1].set_title("high pulse, mucous_membranes")
axs9[1, 2].set_title("high pulse, rectal temperature")

plt.show()

# сравнение данных показывает, что деление на группы невозможно т.к. данные не различаются существенно
# выбираем замену пропусков на средние значения по всем данным

#%%
# Mean или median?
all_horses_df.boxplot(column=['pulse'])
plt.show()

# Распределение скошено, большая часть данных приходится на более маленькие значения,
# в то время как выбросы есть только среди значений превышающие норму для этого показателя в несколько раз
# медиана и среднее будут различаться

# %%
print(
    f"Mean rectal temp: {round(all_horses_df['pulse'].mean(), 3)} \n"
    f"Median rectal temp: {all_horses_df['pulse'].median()} \n"
    f"Mode rectal temp: {all_horses_df['pulse'].mode() }"
)

#%%
# Количество выбросов не такое большое, чтобы повлиять на медиану, поэтому лучшим решением будет
# заполнить пропуски медианным значением

# %%

all_horses_df['pulse'].fillna(all_horses_df['pulse'].median(), inplace=True)

#%%
print(
    f"Mean rectal temp: {round(all_horses_df['pulse'].mean(), 3)} \n"
    f"Median rectal temp: {all_horses_df['pulse'].median()} \n"
    f"Mode rectal temp: {all_horses_df['pulse'].mode() }"
)
#%%
all_horses_df.boxplot(column=['pulse'])
plt.show()

# %% md

### temperature_of_extremities
# ** Значение показателя warm кореллирует с повышенной ректальной температурой. **
# ** Cool и cold c шоком: **
# - ** rectal temperature, ниже 37.8 **
# - ** pulse, выше 30 - 40 **
# - ** mucous membranes, pale pink **
# peripheral pulse: cool - redused, cold - absent, normal - normal, warm - increased

# Пропуски положительно кореллируют с persistalsis, abdominal_distention, mucous_membranes, peripheral_pulse
# посмотрим как взаимосвязаны пропуски
msno.matrix(all_horses_df[['temperature_of_extremities', 'peristalsis',
                           'abdominal_distension', 'peripheral_pulse']])
plt.show()

# Есть данные, где отсутствуют все 4 значения, посмотрим на данные подробнее

#%%
tmp_extrem = all_horses_df[ (all_horses_df['temperature_of_extremities'].isna()) &
               (all_horses_df['peristalsis'].isna()) &
               (all_horses_df['abdominal_distension'].isna()) &
               (all_horses_df['peripheral_pulse'].isna())]
# У записей с пропусками этих данных, много пропусков и по другим данным,
# так что взаимосвязь скорее всего случайна

#%%
# Проверим взаимосвязь значений temperature_of_extremities с другими значениями,
# чтобы проверить можно ли разделить данные на группы для заполнения пропусков

# Значение показателя warm кореллирует с повышенной ректальной температурой
plt.style.use('seaborn-deep')

fig10, ax10 = plt.subplots()
warm_extremities = all_horses_df[ all_horses_df['temperature_of_extremities']=='Warm' ]['rectal_temperature']
norm_extremities = all_horses_df[ all_horses_df['temperature_of_extremities']=='Normal' ]['rectal_temperature']
cold_extremities = all_horses_df[all_horses_df['temperature_of_extremities']=='Cold']['rectal_temperature']
cool_extremities = all_horses_df[all_horses_df['temperature_of_extremities']=='Cool']['rectal_temperature']

ax10.hist([warm_extremities, norm_extremities, cool_extremities, cold_extremities],
          label=['warm_extremities', 'norm_extremities', 'cool_extremities', 'cold_extremities'])
plt.title('Сравнение распредления rectal_temperature')
plt.legend(loc='upper right')
plt.show()

# сравнение распределений показывает, что корреляции между warm temperature_of_extremities
# и повышенной ректальной температурой отсутствует, все выборки распределены примерно одинаково

#%%

plt.style.use('seaborn-deep')

fig11, ax11 = plt.subplots()
warm_extremities = all_horses_df[ all_horses_df['temperature_of_extremities']=='Warm' ]['pulse']
norm_extremities = all_horses_df[ all_horses_df['temperature_of_extremities']=='Normal' ]['pulse']
cold_extremities = all_horses_df[all_horses_df['temperature_of_extremities']=='Cold']['pulse']
cool_extremities = all_horses_df[all_horses_df['temperature_of_extremities']=='Cool']['pulse']

ax11.hist([warm_extremities, norm_extremities, cool_extremities, cold_extremities],
          label=['warm_extremities', 'norm_extremities', 'cool_extremities', 'cold_extremities'])
plt.title('Сравнение распредления pulse относительно температуры конечностей')
plt.legend(loc='upper right')
plt.show()

# взаимосвязь также не прослеживается
#%%
plt.style.use('seaborn-deep')

fig11, axs11 = plt.subplots(1,4)
warm_extremities = all_horses_df[ all_horses_df['temperature_of_extremities']=='Warm' ]['mucous_membranes']\
    .value_counts().plot(kind='bar', ax=axs11[0])
norm_extremities = all_horses_df[ all_horses_df['temperature_of_extremities']=='Normal' ]['mucous_membranes']\
    .value_counts().plot(kind='bar', ax=axs11[1])
cold_extremities = all_horses_df[all_horses_df['temperature_of_extremities']=='Cold']['mucous_membranes']\
    .value_counts().plot(kind='bar', ax=axs11[2])
cool_extremities = all_horses_df[all_horses_df['temperature_of_extremities']=='Cool']['mucous_membranes']\
    .value_counts().plot(kind='bar', ax=axs11[3])

axs11[0].set_title('warm extremeties')
axs11[1].set_title('norm extremeties')
axs11[2].set_title('cold extremeties')
axs11[3].set_title('cool extremeties')

plt.show()

# Если mucous membranes normal_pink, то температура конечностей скорее всего будет norm, warm, cool,
# Если pale_cyanotic, то cool или cold
# dark_cyanotic - cool или cold
# Можно будет использовать эти значения если не будет найдено более четкой группировки по другим

#%%
plt.style.use('seaborn-deep')

fig12, axs12 = plt.subplots(1,4)
warm_extremities = all_horses_df[ all_horses_df['temperature_of_extremities']=='Warm' ]['peripheral_pulse']\
    .value_counts().plot(kind='bar', ax=axs12[0])
norm_extremities = all_horses_df[ all_horses_df['temperature_of_extremities']=='Normal' ]['peripheral_pulse']\
    .value_counts().plot(kind='bar', ax=axs12[1])
cold_extremities = all_horses_df[all_horses_df['temperature_of_extremities']=='Cold']['peripheral_pulse']\
    .value_counts().plot(kind='bar', ax=axs12[2])
cool_extremities = all_horses_df[all_horses_df['temperature_of_extremities']=='Cool']['peripheral_pulse']\
    .value_counts().plot(kind='bar', ax=axs12[3])

axs12[0].set_title('warm extremeties')
axs12[1].set_title('norm extremeties')
axs12[2].set_title('cold extremeties')
axs12[3].set_title('cool extremeties')

plt.show()

# redused - cool или cold
# absent - cool, cold
# normal - normal, warm
# increased - warm, norm, cool

#%%
# Используем значение peripheral pulse для заполнения пропусков

all_horses_df.loc[ (all_horses_df['peripheral_pulse']=='reduced') &
                   (all_horses_df['temperature_of_extremities'].isna()), 'temperature_of_extremities']='Cool'


#%%

all_horses_df.loc[ (all_horses_df['peripheral_pulse']=='absent') &
                   (all_horses_df['temperature_of_extremities'].isna()), 'temperature_of_extremities']='Cold'
#%%

all_horses_df.loc[ (all_horses_df['peripheral_pulse']=='normal') &
                   (all_horses_df['temperature_of_extremities'].isna()), 'temperature_of_extremities']='Normal'
#%%
all_horses_df.loc[ (all_horses_df['peripheral_pulse']=='increased') &
                   (all_horses_df['temperature_of_extremities'].isna()), 'temperature_of_extremities']='Warm'

#%%

all_horses_df['temperature_of_extremities'].isna().sum()

# Осталось 31 незаполненное пропущенное значений, т.к. нехватает данных для определения группы,
# заполним пропущенные значения модой - cool

#%%

all_horses_df['temperature_of_extremities'].fillna('Cool', inplace=True)

#%%
all_horses_df['temperature_of_extremities'].value_counts().plot(kind='bar')
plt.show()

#%% md
### Peristalsis
# peristalsis hypomotile, absent связан с abdominal_distention moderate, severe
# peristalsis absent может быть связан с rectal_examination_feces absent

# Пропуски положительно коррелируют с temperature_of_extremeties, peripheral_pulse, abdominal_distention
# посмотрим как взаимосвязаны пропуски
msno.matrix(all_horses_df[['peristalsis', 'abdominal_distension',
                           'peripheral_pulse']])
plt.show()


perist_nan = all_horses_df[ (all_horses_df['peristalsis'].isna()) &
               (all_horses_df['abdominal_distension'].isna()) &
               (all_horses_df['peripheral_pulse'].isna())]

# здесь ситуция с пропусками та же -
# у записей с пропусками в этих данных много пропусков и по другим данным

# Возможно есть причина пропуска этих данных, но исходя из имеющейся информации, она не очевидна
# MAR

#%%
# Проверим взаимосвязь значений peristalsis с другими значениями,
# чтобы проверить можно ли разделить данные на группы для заполнения пропусков

# Значение показателя peristalsis hypomotile с abdominal_distention (вздутие) moderate, severe

plt.style.use('seaborn-deep')

fig13, ax13 = plt.subplots(1,4)

hypermotile = all_horses_df[ all_horses_df['peristalsis']=='hypermotile' ]['abdominal_distension']\
    .value_counts().plot(kind='bar', ax=ax13[0])
normal = all_horses_df[ all_horses_df['peristalsis']=='normal' ]['abdominal_distension']\
    .value_counts().plot(kind='bar', ax=ax13[1])
hypomotile = all_horses_df[all_horses_df['peristalsis']=='hypomotile']['abdominal_distension']\
    .value_counts().plot(kind='bar', ax=ax13[2])
absent = all_horses_df[all_horses_df['peristalsis']=='absent']['abdominal_distension']\
    .value_counts().plot(kind='bar', ax=ax13[3])

ax13[0].set_title('hypermotile')
ax13[1].set_title('normal')
ax13[2].set_title('hypomotile')
ax13[3].set_title('absent')

plt.show()


# hypermotile = none, slight
# normal = none, slight
# hypomotile = moderate, slight
# absent = severe, moderate

#%%
# Проверим взаимосвязь значений peristalsis с другими значениями,
# чтобы проверить можно ли разделить данные на группы для заполнения пропусков

# Значение показателя peristalsis absent с rectal_examination_feces absent

plt.style.use('seaborn-deep')

fig14, ax14 = plt.subplots(1,4)

hypermotile = all_horses_df[ all_horses_df['peristalsis']=='hypermotile' ]['rectal_examination_feces']\
    .value_counts().plot(kind='bar', ax=ax14[0])
normal = all_horses_df[ all_horses_df['peristalsis']=='normal' ]['rectal_examination_feces']\
    .value_counts().plot(kind='bar', ax=ax14[1])
hypomotile = all_horses_df[all_horses_df['peristalsis']=='hypomotile']['rectal_examination_feces']\
    .value_counts().plot(kind='bar', ax=ax14[2])
absent = all_horses_df[all_horses_df['peristalsis']=='absent']['rectal_examination_feces']\
    .value_counts().plot(kind='bar', ax=ax14[3])

ax14[0].set_title('hypermotile')
ax14[1].set_title('normal')
ax14[2].set_title('hypomotile')
ax14[3].set_title('absent')

plt.show()

# данные сложно интерпертировать, если взаимосвязь между peristalsis и rectal_examination_feces,
# но эти данные сложно использовать для группировки

#%%
# для формирования групп будем использовать группировку по abdominal_distention

all_horses_df.loc[ (all_horses_df['abdominal_distension']=='none') &
                   (all_horses_df['peristalsis'].isna()), 'peristalsis']='hypermotile'

#%%
all_horses_df.loc[ (all_horses_df['abdominal_distension']=='slight') &
                   (all_horses_df['peristalsis'].isna()), 'peristalsis']='normal'

#%%
all_horses_df.loc[ (all_horses_df['abdominal_distension']=='moderate') &
                   (all_horses_df['peristalsis'].isna()), 'peristalsis']='hypomotile'

#%%
all_horses_df.loc[ (all_horses_df['abdominal_distension']=='severe') &
                   (all_horses_df['peristalsis'].isna()), 'peristalsis']='absent'

#%%

all_horses_df['peristalsis'].isna().sum()

# Осталось 22 незаполненных значения, там, где abdominal_distension так же имеет пропуски,
# заполним пропущенные значения модой - hypomotile
#%%

all_horses_df['peristalsis'].fillna('hypomotile', inplace=True)

#%%
all_horses_df['peristalsis'].value_counts().plot(kind='bar')
plt.show()

#%% md
### Abdomen
# abdomen firm_feces_large_intestine может быть связан с rectal_examination_feces absent
# и nasogastric_reflux больше 1 литра.
# abdomen distended small intestine и distended large intestine с хирургическими осложнениями -
# первая цифра 2 = sm intestine или  3 = lg colon, четвертая цифра 1 = obstruction

# пропуски незначительно кореллируют с rectal_examination_feces

msno.matrix(all_horses_df[['abdomen', 'rectal_examination_feces']])
plt.show()


abdomen_nan = all_horses_df[ (all_horses_df['abdomen'].isna()) &
               (all_horses_df['rectal_examination_feces'].isna())]

# Взаимосвязь пропусков выглядит случайной, т.к. в Abdomen и rectal_examination_feces много пропусков (> 30%)
# Более того, в записях с пропущенными Abdomen и rectal_examination_feces много пропусков и по другим данным

# Причина пропусков не ясна, возможно это не обязательное обследование
# и для принятия нужного решения достаточно набора других данных
# MAR


#%%
# Проверим взаимосвязь значений abdomen с другими значениями,
# чтобы проверить можно ли разделить данные на группы для заполнения пропусков

# abdomen firm_feces_large_intestine может быть связан с rectal_examination_feces absent

plt.style.use('seaborn-deep')

fig15, ax15 = plt.subplots(1,5, figsize=(16, 10))

abdomen_normal = all_horses_df[ all_horses_df['abdomen']=='normal' ]['rectal_examination_feces']\
    .value_counts().plot(kind='bar', ax=ax15[0])
abdomen_other = all_horses_df[ all_horses_df['abdomen']=='other' ]['rectal_examination_feces']\
    .value_counts().plot(kind='bar', ax=ax15[1])
firm_feces_large_intestine = all_horses_df[all_horses_df['abdomen']=='firm_feces_large_intestine']['rectal_examination_feces']\
    .value_counts().plot(kind='bar', ax=ax15[2])
distended_small_intestine = all_horses_df[all_horses_df['abdomen']=='distended_small_intestine']['rectal_examination_feces']\
    .value_counts().plot(kind='bar', ax=ax15[3])
distended_large_intestine = all_horses_df[all_horses_df['abdomen']=='distended_large_intestine']['rectal_examination_feces']\
    .value_counts().plot(kind='bar', ax=ax15[4])

ax15[0].set_title('abdomen_normal')
ax15[1].set_title('abdomen_other')
ax15[2].set_title('firm_feces_large_intestine')
ax15[3].set_title('distended_small_intestine')
ax15[4].set_title('distended_large_intestine')


plt.show()

# normal feces = abdomen normal
# decreased нет четкой взаимосвязи
# increased нет четкой взаимосвязи
# absent = distended_large_intestine, distended_small_intestine


#%%

# abdomen firm_feces_large_intestine и nasogastric_reflux больше 1 литра

plt.style.use('seaborn-deep')

fig16, ax16 = plt.subplots(1,5, figsize=(16, 10))

abdomen_normal = all_horses_df[ all_horses_df['abdomen']=='normal' ]['nasogastric_reflux']\
    .value_counts().plot(kind='bar', ax=ax16[0])
abdomen_other = all_horses_df[ all_horses_df['abdomen']=='other' ]['nasogastric_reflux']\
    .value_counts().plot(kind='bar', ax=ax16[1])
firm_feces_large_intestine = all_horses_df[all_horses_df['abdomen']=='firm_feces_large_intestine']['nasogastric_reflux']\
    .value_counts().plot(kind='bar', ax=ax16[2])
distended_small_intestine = all_horses_df[all_horses_df['abdomen']=='distended_small_intestine']['nasogastric_reflux']\
    .value_counts().plot(kind='bar', ax=ax16[3])
distended_large_intestine = all_horses_df[all_horses_df['abdomen']=='distended_large_intestine']['nasogastric_reflux']\
    .value_counts().plot(kind='bar', ax=ax16[4])

ax16[0].set_title('abdomen_normal')
ax16[1].set_title('abdomen_other')
ax16[2].set_title('firm_feces_large_intestine')
ax16[3].set_title('distended_small_intestine')
ax16[4].set_title('distended_large_intestine')


plt.show()

# взаимосвязь между данными не проследивается

#%%
 # abdomen distended small intestine и distended large intestine с хирургическими осложнениями

plt.style.use('seaborn-deep')

fig17, ax17 = plt.subplots(1,5, figsize=(16, 10))

abdomen_normal = all_horses_df[ all_horses_df['abdomen']=='normal' ]['surgical_lesion']\
    .value_counts().plot(kind='bar', ax=ax17[0])
abdomen_other = all_horses_df[ all_horses_df['abdomen']=='other' ]['surgical_lesion']\
    .value_counts().plot(kind='bar', ax=ax17[1])
firm_feces_large_intestine = all_horses_df[all_horses_df['abdomen']=='firm_feces_large_intestine']['surgical_lesion']\
    .value_counts().plot(kind='bar', ax=ax17[2])
distended_small_intestine = all_horses_df[all_horses_df['abdomen']=='distended_small_intestine']['surgical_lesion']\
    .value_counts().plot(kind='bar', ax=ax17[3])
distended_large_intestine = all_horses_df[all_horses_df['abdomen']=='distended_large_intestine']['surgical_lesion']\
    .value_counts().plot(kind='bar', ax=ax17[4])

ax17[0].set_title('abdomen_normal')
ax17[1].set_title('abdomen_other')
ax17[2].set_title('firm_feces_large_intestine')
ax17[3].set_title('distended_small_intestine')
ax17[4].set_title('distended_large_intestine')


plt.show()

# Есть взаимосвяязь между отсутствием хирургических осложнений и - abdomen normal,  abdomen other и firm feces large intestine
# Также как и наличием хирургических осложнений и distended large intestine, distended small intestine
# посмотрим побробнее на конкретные осложнения
#%%
plt.style.use('seaborn-deep')

fig18, ax18 = plt.subplots(1,5, figsize=(16, 10))

abdomen_normal = all_horses_df[ (all_horses_df['abdomen']=='normal') &
                                (all_horses_df['surgical_lesion']=='Yes') ]['type_of_lesion_1']\
.value_counts().plot(kind='bar', ax=ax18[0])

abdomen_other = all_horses_df[ (all_horses_df['abdomen']=='other') &
                                (all_horses_df['surgical_lesion']=='Yes') ]['type_of_lesion_1']\
.value_counts().plot(kind='bar', ax=ax18[1])

firm_feces_large_intestine = all_horses_df[ (all_horses_df['abdomen']=='firm_feces_large_intestine') &
                                (all_horses_df['surgical_lesion']=='Yes') ]['type_of_lesion_1']\
.value_counts().plot(kind='bar', ax=ax18[2])

distended_small_intestine = all_horses_df[ (all_horses_df['abdomen']=='distended_small_intestine') &
                                (all_horses_df['surgical_lesion']=='Yes') ]['type_of_lesion_1']\
.value_counts().plot(kind='bar', ax=ax18[3])

distended_large_intestine = all_horses_df[ (all_horses_df['abdomen']=='distended_large_intestine') &
                                (all_horses_df['surgical_lesion']=='Yes') ]['type_of_lesion_1']\
.value_counts().plot(kind='bar', ax=ax18[4])


ax18[0].set_title('abdomen_normal')
ax18[1].set_title('abdomen_other')
ax18[2].set_title('firm_feces_large_intestine')
ax18[3].set_title('distended_small_intestine')
ax18[4].set_title('distended_large_intestine')

plt.show()

#%%
# Получим список осложнений, который появляется только при определенных значениях abdomen

normal_abdomen_lesion = set(all_horses_df[
    (all_horses_df['abdomen']=='normal') &
    (all_horses_df['surgical_lesion']=='Yes') ]['type_of_lesion_1'].unique())

abdomen_other_lesion = set(all_horses_df[
    (all_horses_df['abdomen']=='other') &
    (all_horses_df['surgical_lesion']=='Yes') ]['type_of_lesion_1'].unique())

firm_feces_large_intestine_lesion = set(all_horses_df[
    (all_horses_df['abdomen']=='firm_feces_large_intestine') &
    (all_horses_df['surgical_lesion']=='Yes') ]['type_of_lesion_1'].unique())

distended_small_intestine_lesion = set(all_horses_df[
    (all_horses_df['abdomen']=='distended_small_intestine') &
    (all_horses_df['surgical_lesion']=='Yes') ]['type_of_lesion_1'].unique())

distended_large_intestine_lesion = set(all_horses_df[
    (all_horses_df['abdomen']=='distended_large_intestine') &
    (all_horses_df['surgical_lesion']=='Yes') ]['type_of_lesion_1'].unique())

#%%
normal_abdomen_lesion_only = (normal_abdomen_lesion - abdomen_other_lesion -
                              firm_feces_large_intestine_lesion - distended_small_intestine_lesion -
                              distended_large_intestine_lesion)

abdomen_other_lesion_only = (abdomen_other_lesion - normal_abdomen_lesion -
                             firm_feces_large_intestine_lesion - distended_small_intestine_lesion -
                              distended_large_intestine_lesion)

firm_feces_large_intestine_lesion_only = (firm_feces_large_intestine_lesion - normal_abdomen_lesion -
                                          abdomen_other_lesion - distended_small_intestine_lesion -
                                          distended_large_intestine_lesion)

distended_small_intestine_lesion_only = (distended_small_intestine_lesion - firm_feces_large_intestine_lesion -
                                         normal_abdomen_lesion - abdomen_other_lesion -
                                         distended_large_intestine_lesion)

distended_large_intestine_lesion_only = (distended_large_intestine_lesion - distended_small_intestine_lesion -
                                         firm_feces_large_intestine_lesion - normal_abdomen_lesion -
                                         abdomen_other_lesion)

print(
    f"Normal abdomen: {normal_abdomen_lesion_only} \n"
    f"Other abdomen: {abdomen_other_lesion_only} \n"
    f"Firm_feces_large_intestine: {firm_feces_large_intestine_lesion_only} \n"
    f"Distended_small_intestine: {distended_small_intestine_lesion_only} \n"
    f"Distended_large_intestine: {distended_large_intestine_lesion_only} \n"
)

# Для normal, other abdomen и Distended_large_intestine взаимосвязь с типами осложенией скорее всего случайна,
# что следует из количества данных и описания,
# но для заполнения пропусков для Distended_small_intestine и Distended_large_intestine эти данные можно использовать

# %%
all_horses_df.loc[ (all_horses_df['type_of_lesion_1'].isin(distended_small_intestine_lesion_only)) &
                   (all_horses_df['abdomen'].isna()), 'abdomen']='distended_small_intestine'

# %%
all_horses_df.loc[ (all_horses_df['type_of_lesion_1'].isin(distended_large_intestine_lesion_only)) &
                   (all_horses_df['abdomen'].isna()), 'abdomen']='distended_large_intestine'

#%%

all_horses_df['abdomen'].isna().sum()

# Осталось 87 незаполненное пропущенное значений. заполним их с помощью значений rectal_examination_feces

#%%
all_horses_df.loc[ (all_horses_df['rectal_examination_feces']=='normal' ) &
                   (all_horses_df['abdomen'].isna()), 'abdomen']='normal'
#%%
all_horses_df.loc[ (all_horses_df['rectal_examination_feces']=='absent' ) &
                   (all_horses_df['abdomen'].isna()), 'abdomen']='distended_large_intestine'

#%%
all_horses_df['abdomen'].isna().sum()

# Осталось 61 незаполненное значение, заполнение любым средним значением слишком сильно изменит распределение данных,
# но у нас недостаточно данных для группировки и заполнения пропусков средним/модой по группе
# удаление 5 части значений одинаково плохо отразится на данных, как и замена пропусков модой
# в такой ситуации логичнее всего принимать решение исходя из целей дальнейшей работы с этим датасетом,
# но т.к. у нас нет этой информации, будем считать, что в приоритете сохранить наибольшее количество данных

#%%
# Заполним оставшиеся пропуски модой - distended_large_intestine

all_horses_df['abdomen'].fillna('distended_large_intestine')

#%%
all_horses_df['abdomen'].value_counts().plot(kind='bar')
plt.show()

# %%

# Сформируем датасет без пропусков

horses_df = all_horses_df.dropna(axis='columns', how='any')

#%%
horses_df.info()