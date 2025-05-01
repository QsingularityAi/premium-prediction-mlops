# Kubernetes Deployment for Premium Prediction MLOps

This document explains the Kubernetes configuration files used to deploy the Premium Prediction MLOps application.

## Overview

The Kubernetes setup deploys the application as a set of services within a dedicated namespace. It includes configurations for the application deployment, service exposure, configuration management, secrets handling, and monitoring.

## Files

The `kubernetes/` directory contains the following configuration files:

-   **`namespace.yaml`**: Defines the Kubernetes namespace (`premium-prediction`) where all application resources will reside. This helps in organizing and isolating the application components.
-   **`configmap.yaml`**: Contains non-sensitive configuration data for the application, such as environment variables or configuration files that the application needs to run.
-   **`secrets.yaml`**: Manages sensitive information like API keys, database passwords, or other credentials. These are stored securely within the cluster. *Note: Ensure this file is properly secured and potentially managed using a secrets management tool like HashiCorp Vault or Sealed Secrets in production.*
-   **`deployment.yaml`**: Defines the application's deployment strategy. It specifies the Docker image to use, the number of replicas (pods), resource requests/limits, and how updates should be rolled out.
-   **`service.yaml`**: Exposes the application deployment as a network service within the cluster. It defines how other services within the cluster can reach the application pods (e.g., using a ClusterIP).
-   **`ingress.yaml`**: Manages external access to the services within the cluster, typically HTTP/HTTPS. It defines rules for routing external traffic to the appropriate services based on hostnames or paths. This requires an Ingress controller to be running in the cluster.
-   **`monitoring.yaml`**: Sets up monitoring components. This might include configurations for Prometheus to scrape metrics from the application, potentially deploying Prometheus Operator, Grafana dashboards, or Alertmanager rules. *(The presence of `monitoring.yaml.backup` suggests previous configurations or changes.)*
-   **`local-pv.yaml` & `local-grafana-pv.yaml`**: These files (located in the root directory, but related to Kubernetes persistence) likely define Persistent Volumes (PVs) for local storage, possibly for development or testing environments where standard cloud provider storage is not available or desired. `local-grafana-pv.yaml` specifically seems intended for Grafana persistence.

## Deployment Steps

1.  **Ensure a Kubernetes cluster is running and `kubectl` is configured.**
2.  **Create the namespace:**
    ```bash
    kubectl apply -f kubernetes/namespace.yaml
    ```
3.  **Create the ConfigMap:**
    ```bash
    kubectl apply -f kubernetes/configmap.yaml -n premium-prediction
    ```
4.  **Create the Secrets:**
    *Ensure your `secrets.yaml` contains the actual base64 encoded secrets before applying.*
    ```bash
    kubectl apply -f kubernetes/secrets.yaml -n premium-prediction
    ```
5.  **Deploy the application:**
    ```bash
    kubectl apply -f kubernetes/deployment.yaml -n premium-prediction
    ```
6.  **Expose the application service:**
    ```bash
    kubectl apply -f kubernetes/service.yaml -n premium-prediction
    ```
7.  **Configure Ingress (if an Ingress controller is set up):**
    ```bash
    kubectl apply -f kubernetes/ingress.yaml -n premium-prediction
    ```
8.  **Set up Monitoring:**
    ```bash
    kubectl apply -f kubernetes/monitoring.yaml -n premium-prediction
    ```

## Accessing the Application

-   **Internal Access:** Services within the cluster can access the application via its service name (e.g., `premium-prediction-service.premium-prediction.svc.cluster.local`).
-   **External Access:** If Ingress is configured, the application should be accessible via the hostname defined in `ingress.yaml`. You might need to configure DNS or use port-forwarding for local testing:
    ```bash
    kubectl port-forward svc/premium-prediction-service -n premium-prediction 8080:<containerPort>
    ```
    Replace `<containerPort>` with the port your application listens on inside the container.

## Monitoring

If Prometheus and Grafana are set up via `monitoring.yaml`, you can access Grafana to view dashboards related to the application's performance and metrics. Access methods depend on how Grafana is exposed (Service, Ingress, Port-forwarding).
