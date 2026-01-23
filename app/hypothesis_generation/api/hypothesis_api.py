from flask import Blueprint, request, jsonify
from app.storage.mongo_storage import mongo_db_manager
import uuid
import datetime

hypothesis_bp = Blueprint('hypothesis_api', __name__)

@hypothesis_bp.route('/enrich', methods=['POST'])
def start_enrichment():
    """
    Step 1: Start Enrichment
    Input: variant, project_id, tissue_name
    Output: hypothesis_id (202 Accepted)
    """
    try:
        data = request.json
        variant = data.get('variant')
        project_id = data.get('project_id')
        tissue_name = data.get('tissue_name')

        if not all([variant, project_id, tissue_name]):
             return jsonify({"error": "Missing required fields"}), 400

        # Generate a hypothesis ID
        hypothesis_id = f"hyp_{uuid.uuid4().hex[:8]}"
        
        # Store the request in MongoDB with 'pending' status
        request_data = {
            "id": hypothesis_id,
            "variant": variant,
            "project_id": project_id,
            "tissue_name": tissue_name,
            "status": "pending",
            "created_at": datetime.datetime.utcnow(),
            "enrich_id": None,
            "task_history": []
        }
        mongo_db_manager.create_hypothesis_request(request_data)

        return jsonify({
            "hypothesis_id": hypothesis_id,
            "project_id": project_id
        }), 202

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@hypothesis_bp.route('/hypothesis', methods=['GET'])
def check_hypothesis_status():
    """
    Step 2: Check Hypothesis Status
    Input: id (hypothesis_id)
    Output: Status JSON (Pending or Completed)
    """
    try:
        hypothesis_id = request.args.get('id')
        if not hypothesis_id:
            return jsonify({"error": "Missing id parameter"}), 400

        # Retrieve request from MongoDB
        request_data = mongo_db_manager.get_hypothesis_request(hypothesis_id)
        if not request_data:
            return jsonify({"error": "Hypothesis ID not found"}), 404

        # Simulation Logic: If created > 10 seconds ago, mark as completed
        created_at = request_data.get('created_at')
        if request_data.get('status') == 'pending':
            time_diff = (datetime.datetime.utcnow() - created_at).total_seconds()
            if time_diff > 10:
                # Update to completed
                enrich_id = f"enrich_{uuid.uuid4().hex[:8]}"
                mongo_db_manager.update_hypothesis_status(hypothesis_id, "completed", enrich_id)
                request_data['status'] = 'completed'
                request_data['enrich_id'] = enrich_id

        # Format response
        response = {
            "id": request_data.get('id'),
            "variant": request_data.get('variant'),
            "phenotype": "Obesity", # Hardcoded as per requirements/mock
            "status": request_data.get('status'),
            "created_at": request_data.get('created_at').isoformat() + "Z",
            "task_history": request_data.get('task_history', []),
            "enrich_id": request_data.get('enrich_id')
        }

        return jsonify(response), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@hypothesis_bp.route('/enrich', methods=['GET'])
def get_enrichment_results():
    """
    Step 3: Retrieve Enrichment Results
    Input: id (enrich_id)
    Output: Mocked biological data
    """
    try:
        enrich_id = request.args.get('id')
        if not enrich_id:
            return jsonify({"error": "Missing id parameter"}), 400

        # In a real app, we'd look up the enrich_id. 
        # For this mock, we return the hardcoded data as requested.
        
        response = {
            "id": enrich_id,
            "variant": "rs1421085", # Mocked consistency
            "phenotype": "Obesity",
            "causal_gene": "FTO",
            "GO_terms": [
                {
                    "id": "GO:1904177",
                    "name": "Regulation of Adipose Tissue Development",
                    "genes": ["PARP1", "PLAAT3", "PPARG"],
                    "p": 0.00495,
                    "rank": 1
                },
                {
                    "id": "GO:0045598",
                    "name": "Regulation of Fat Cell Differentiation",
                    "genes": ["CEBPB", "ADIPOQ", "PPARG"],
                    "p": 0.0266,
                    "rank": 2
                }
            ]
        }
        return jsonify(response), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@hypothesis_bp.route('/hypothesis', methods=['POST'])
def generate_hypothesis_final():
    """
    Step 4: Generate Hypothesis
    Input: id (enrich_id), go (GO term ID)
    Output: Summary and Graph
    """
    try:
        data = request.json
        enrich_id = data.get('id')
        go_term = data.get('go')

        if not all([enrich_id, go_term]):
             return jsonify({"error": "Missing required fields"}), 400

        # Return the hardcoded mocked response
        response = {
            "summary": "The variant rs1421085 may influence obesity by altering regulation of adipose tissue development through the FTO gene.",
            "graph": {
                "nodes": [
                    { "id": "rs1421085", "type": "snp", "name": "rs1421085" },
                    { "id": "FTO", "type": "gene", "name": "FTO" },
                    { "id": "GO:1904177", "type": "go", "name": "Regulation of Adipose Tissue Development" }
                ],
                "edges": [
                    { "source": "rs1421085", "target": "FTO", "label": "affects" },
                    { "source": "FTO", "target": "GO:1904177", "label": "involved_in" }
                ],
                "probability": 0.95
            }
        }
        return jsonify(response), 201

    except Exception as e:
        return jsonify({"error": str(e)}), 500
