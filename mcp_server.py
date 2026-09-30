import sys
import json
from client import PPOClippedLossEvaluator

def handle_request(req):
    method = req.get("method")
    req_id = req.get("id")
    
    if method == "initialize":
        return {
            "jsonrpc": "2.0",
            "id": req_id,
            "result": {
                "protocolVersion": "2024-11-05",
                "capabilities": {"tools": {}},
                "serverInfo": {"name": "genpark-ppo-clipped-surrogate-loss-evaluator-skill", "version": "1.0.0"}
            }
        }
    elif method == "tools/list":
        return {
            "jsonrpc": "2.0",
            "id": req_id,
            "result": {
                "tools": [
                    {
                        "name": "calculate_ppo_loss",
                        "description": "Calculates PPO clipped surrogate objective and KL penalty",
                        "inputSchema": {
                            "type": "object",
                            "properties": {
                                "logprob_new": {"type": "number"},
                                "logprob_old": {"type": "number"},
                                "advantage": {"type": "number"},
                                "epsilon": {"type": "number", "default": 0.2}
                            },
                            "required": ["logprob_new", "logprob_old", "advantage"]
                        }
                    }
                ]
            }
        }
    elif method == "tools/call":
        params = req.get("params", {})
        name = params.get("name")
        args = params.get("arguments", {})
        
        if name == "calculate_ppo_loss":
            res = PPOClippedLossEvaluator.evaluate_clipped_loss(
                args.get("logprob_new", 0.0),
                args.get("logprob_old", 0.0),
                args.get("advantage", 0.0),
                args.get("epsilon", 0.2)
            )
            return {"jsonrpc": "2.0", "id": req_id, "result": {"content": [{"type": "text", "text": json.dumps(res, indent=2)}]}}
            
    return {"jsonrpc": "2.0", "id": req_id, "error": {"code": -32601, "message": "Method not found"}}

def run():
    for line in sys.stdin:
        if not line.strip():
            continue
        try:
            req = json.loads(line)
            res = handle_request(req)
            sys.stdout.write(json.dumps(res) + "\n")
            sys.stdout.flush()
        except Exception as e:
            sys.stdout.write(json.dumps({"jsonrpc": "2.0", "error": {"code": -32000, "message": str(e)}}) + "\n")
            sys.stdout.flush()

if __name__ == "__main__":
    run()
