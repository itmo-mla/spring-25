# Лабораторная работа №4. Латентное размещение Дирихле

Был использован набор данных https://www.kaggle.com/datasets/gondimalladeepesh/nvidia-documentation-question-and-answer-pairs. В нём хранятся текста Q&A секции документации Nvidia.

Сравнивал свою реализацию LDA и библиотечную версию `sklearn.decomposition.LatentDirichletAllocation`.

Результат применения библиотечной версии:
```
Тема 1: windows, cuda, nvidia, learning, ai, parallel, deep, platforms, linux, altimesh
Тема 2: cuda, learning, facial, deep, researchers, recognition, cudnn, nvidia, gpus, real
Тема 3: wonder, bot, cuda, gpus, information, using, titan, project, face2face, facial
Тема 4: using, fraudoscope, gpus, code, does, detecting, cuda, com, jet, lie
Тема 5: cuda, gpus, robot, learning, researchers, purpose, deep, university, kernel, vectoradd
Тема 6: cuda, nvidia, gpus, systems, arm, like, library, tools, researchers, parallel
Тема 7: developed, social, researchers, cuda, robot, gpu, learning, used, request, capable
Тема 8: jetpack, cuda, scientists, deep, computing, learning, workloads, ai, k40, developed
Тема 9: gpu, gpus, using, nvidia, code, hybridizer, cuda, accelerated, eddy, tool
Тема 10: cuda, learning, deep, ai, gpu, developers, ngc, collection, purpose, frameworks
```

Результат моего алгоритма:
```
Topic 1: the, to, of, time, video
Topic 2: the, on, for, cuda, library
Topic 3: fraudoscope, does, facial, detecting, to
Topic 4: earthquake, accelerated, s, california, about
Topic 5: and, code, to, nvidia, visual
Topic 6: the, is, of, with, eddy
Topic 7: cuda, is, the, of, and
Topic 8: the, in, is, of, parallel
Topic 9: for, in, com, jet, and
Topic 10: and, the, gpus, learning, cuda
```

Сравнение метрик:
```
C_V Coherence (higher is better):
Custom LDA: 0.7848
Scikit-learn LDA: 0.7072
Result: Scikit-learn LDA has higher c_v coherence

U_MASS Coherence (higher is better):
Custom LDA: 0.1744
Scikit-learn LDA: 0.1578
Result: Scikit-learn LDA has higher u_mass coherence

C_UCI Coherence (higher is better):
Custom LDA: 0.4966
Scikit-learn LDA: 0.4789
Result: Scikit-learn LDA has higher c_uci coherence
```

Из экспериментов делаю вывод о том, что моя реализация рабочая и сопоставима с библиотечной версией по качеству (однако последняя всё же лучше). Скорость обучения сопоставима (оба учатся порядка минуты).