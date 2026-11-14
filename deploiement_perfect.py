
cat << 'EOF' > deploiement_prefect.py
from prefect import serve
from pipeline_prefect import flow_all, flow_entrainement, flow_evaluate

if __name__ == "__main__":
    all_deploy = flow_all.to_deployment(
        name="ml-pipeline-all",
        cron="0 2 * * *",
        tags=["full-pipeline", "mlops"]
    )

    train_deploy = flow_entrainement.to_deployment(
        name="ml-pipeline-train",
        tags=["training", "mlops"]
    )

    evaluate_deploy = flow_evaluate.to_deployment(
        name="ml-pipeline-evaluate",
        tags=["evaluation", "mlops"]
    )

    serve(all_deploy, train_deploy, evaluate_deploy)
