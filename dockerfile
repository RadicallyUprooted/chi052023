FROM python:3.9

WORKDIR /app

COPY requirements.txt .

RUN pip install -r requirements.txt

COPY train.py .
COPY inference.py .
COPY model.pth .

COPY test_data test_data

CMD ["python", "inference.py", "--input", "test_data"]