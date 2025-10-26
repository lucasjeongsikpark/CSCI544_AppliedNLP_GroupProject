# 🧠 LLaMA3 API Deployment Guide

## [Server: CARC]

```bash
export HF_TOKEN={your_huggingface_token}
vi run_model.sh   # copy and paste run_model.sh content
sbatch run_model.sh {JOB_NAME} {MODEL_ID} {PORT_NUMBER}
# Example
sbatch run_model.sh llama meta-llama/Meta-Llama-3.1-8B-Instruct 8080
```

> 💡 You may have to sign in to Hugging Face to load the model for the first time.

### Argument Details

| Argument        | Description                    | Example                                 |
| --------------- | ------------------------------ | --------------------------------------- |
| **JOB_NAME**    | Any name you want for your job | `llama3_api`                            |
| **MODEL_ID**    | Model name from Hugging Face   | `meta-llama/Meta-Llama-3.1-8B-Instruct` |
| **PORT_NUMBER** | Port number to serve the API   | `8080`                                  |

⏳ **Note:**  
If it's your first time loading the model, please wait about **30 minutes**.  
You can check the health and progress logs in the `logs/` directory.

---

## [Client]

In your **local terminal**, run the following command so that you can **create an SSH tunnel to the CARC server**.  
You can check **NODE_NAME (Node List)** here: [CARC Active Jobs Dashboard](https://ondemand.carc.usc.edu/pun/sys/dashboard/activejobs)

```bash
./connect_carc_api.sh {NODE_NAME} {PORT_NUMBER} {USC_EMAIL_PREFIX}
# Example
./connect_carc_api.sh a04-20 8080 jeongsik
```

Then, open **another terminal session** to **call the model and perform inference**:

```bash
python client_call_llm.py
```
