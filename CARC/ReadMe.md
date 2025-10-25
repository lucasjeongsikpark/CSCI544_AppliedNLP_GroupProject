[Client]
./connect_carc_api.sh a04-20 8081 jeongsik
in another session (terminal)
python client_call_llm.py

[CARC]
sbatch llama3_api.sh (wait for 30 min if it's your first time loading the model)
