from datetime import datetime

from airflow import DAG
from airflow.operators.python import PythonOperator

from src.main import prepare_data , train , evaluate

with DAG(
    dag_id = "student_mental_health_pipeline",
    description = "Student Mental Health Risk Prediction Pipeline",
    start_date = datetime(2026 , 11 , 14),
    schedule = "@daily",
    catchup = False,
    tags = ["Machine Learning", "Mental Health"]
) as dag:
    prepare_task = PythonOperator(
        task_id = "prepare_data",
        python_callable = prepare_data
    )

    train_task = PythonOperator(
        task_id = "train",
        python_callable = train
    )

    evaluate_task = PythonOperator(
        task_id = "evaluate",
        python_callable = evaluate
    )

    prepare_task >> train_task >> evaluate_task