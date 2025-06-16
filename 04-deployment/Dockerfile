FROM agrigorev/zoomcamp-model:mlops-2024-3.10.13-slim

RUN pip install -U pip
RUN pip install pipenv

COPY [ "Pipfile", "Pipfile.lock", "./" ]

# If I exclude the following line and use the model in the provided docker image,
# the results are vastly different -- all mean duraitons are close to 19 which fits 
# neither of the answers in Q5 and Q6. 
COPY ["model.bin", "model.bin"]

COPY ["homework.py", "./"]

COPY ["output/", "output/"]

RUN pipenv install --system --deploy

ENTRYPOINT  ["python", "homework.py", "2023", "5"]