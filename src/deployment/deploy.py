from mlflow.deployments import get_deploy_client

client = get_deploy_client("databricks")
endpoint = client.create_endpoint(
    name="nbapred",
    config={
        "served_models": [
            {
                "model_name": "NBA_XGB_Final",
                "model_version": "1",
                "workload_size": "Medium",
                "scale_to_zero_enabled": True,
                "compute_type": "CPU"
            }
        ],
        "traffic_config": {
            "routes": [
                {
                    "served_model_name": "NBA_XGB_Final-1",
                    "traffic_percentage": 100
                }
            ]
        },
        "enable_route_optimization": True
    }
)

print(f"Endpoint '{endpoint.name}' created successfully.")
print(f"Endpoint URL: {endpoint.url}")