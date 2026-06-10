import copy
import json
import logging
import os
from dotenv import load_dotenv
from app.annotation_graph.neo4j_handler import Neo4jConnection
from app.annotation_graph.schema_handler import SchemaHandler
from app.llm_handle.llm_models import LLMInterface
from app.prompts.annotation_prompts import (
    EXTRACT_RELEVANT_INFORMATION_PROMPT,
    JSON_CONVERSION_PROMPT,
    ORGANISM_DETECTION_PROMPT,
    SELECT_PROPERTY_VALUE_PROMPT,
    SELECT_PROPERTY_VALUES_BATCH_PROMPT,
    RESULT_SUMMARIZATION_PROMPT,
)
from app.socket_manager import emit_to_user
from app.storage.redis import redis_manager
from .json_to_cypher import JsonToCypherConverter


logger = logging.getLogger(__name__)

load_dotenv()


class Graph:
    def __init__(self, llm: LLMInterface, schema_handler: SchemaHandler, fly_schema_handler: SchemaHandler = None) -> None:
        self.llm = llm
        self.schema_handler = schema_handler
        self.enhanced_schema = schema_handler.enhanced_schema
        self.neo4j = Neo4jConnection(
            uri=os.getenv("NEO4J_URI"),
            username=os.getenv("NEO4J_USERNAME"),
            password=os.getenv("NEO4J_PASSWORD"),
        )

        # Fly organism resources (schema + separate Neo4j connection)
        self.fly_schema_handler = fly_schema_handler
        self.fly_enhanced_schema = fly_schema_handler.enhanced_schema if fly_schema_handler else None
        self.fly_neo4j = Neo4jConnection(
            uri=os.getenv("FLY_NEO4J_URI"),
            username=os.getenv("FLY_NEO4J_USERNAME"),
            password=os.getenv("FLY_NEO4J_PASSWORD"),
        ) if os.getenv("FLY_NEO4J_URI") else None
        # Pending annotation confirmations — Redis-backed (cross-process), in-memory fallback
        self._redis = redis_manager
        self._pending_fallback: dict = {}  # used only when Redis is unavailable
        self._PENDING_TTL = 600  # 10 minutes

        # Maps node type → the Neo4j property to use when searching by the JSON `id` field
        self._node_id_property = {
            "gene": "gene_name",
            "transcript": "transcript_id",
            "exon": "exon_id",
            "protein": "protein_id",
            "variant": "variant_id",
            "pathway": "pathway_name",
            "go_term": "go_id",
            "tad": "tad_id",
            "regulatory_element": "regulatory_element_id",
            "enhancer": "enhancer_id",
            "promoter": "promoter_id",
        }

    def query_knowledge_graph(self, json_query, token):
        """
        Query the knowledge graph service.

        Args:
            json_query (dict): The JSON query to be sent.

        Returns:
            dict: The JSON response from the knowledge graph service or an error message.
        """
        if isinstance(json_query, str):
            logger.info("passed json is a string changing it to a dicitionary")
            json_query = json.loads(json_query)

        logger.info("Starting knowledge graph query...")
        source = "ai-assistant"
        limit = 100

        params = {"source": source, "limit": limit, "properties": True}
        payload = {"requests": json_query}

        try:
            logger.debug(
                f"Sending request to {self.kg_service_url} with payload: {payload}"
            )
            response = requests.post(
                self.kg_service_url + "/query",
                json=payload,
                params=params,
                headers={"Authorization": f"Bearer {token}"},
            )
            response.raise_for_status()
            return response.json()
        except requests.RequestException as e:
            logger.error(f"Error querying knowledge graph: {e}")
            if e.response is not None:
                logger.error(f"Response content: {e.response.text}")
            return {"error": f"Failed to query knowledge graph: {str(e)}"}

    def validated_json(self, query, user_id):
        logger.info(f"Starting annotation query processing for question: '{query}'")

        # Extract relevant information
        relevant_information = self._extract_relevant_information(query)

        # Convert to initial JSON
        emit_to_user(user=user_id, message=f"Validating Constructed Json Format...")
        initial_json = self._convert_to_annotation_json(relevant_information, query)

        # Validate and update
        validation = self._validate_and_update(initial_json)

        # If validation failed, return the intermediate steps
        if validation["validation_report"]["validation_status"] == "failed":
            logger.error("Validation is failing *****sending the intial json format")
            return {
                "text": None,
                "json_format": initial_json,
                "organism": "human",
                "resource": {"id": None, "type": "annotation"},
            }

        validated_json = validation["updated_json"]
        return {
            "text": None,
            "json_format": validated_json,
            "organism": "human",
            "resource": {"id": None, "type": "annotation"},
        }

    def generate_graph(self, validated_json, token):
        try:
            graph = self.query_knowledge_graph(validated_json, token)



            response = {
                "text": graph["answer"],
                "resource": {"id": graph["annotation_id"], "type": "annotation"},
            }
            # Store summary in Redis cache for 24 hours
            # redis_manager.create_graph(graph_id=graph_id, graph_summary=summary_text)

            logger.info("Completed query processing.")
            return response

        except Exception as e:
            logger.error(f"An error occurred during graph generation: {e}")
            return {
                "text": f"I apologize, but I wasn't able to generate the graph you requested. Could you please rephrase your question or provide additional details so I can better understand what you're looking for?"
            }

    # Keyword signals that unambiguously indicate Drosophila — checked before LLM call.
    # Two-pronged fly detection:
    # 1. Explicit schema request ("fly schema", "using fly", "drosophila schema")
    # 2. Unambiguous fly-specific biology (anatomy, identifiers, species names)
    # Ambiguous gene names (brca2, p53) are NOT signals — users will say "fly schema" explicitly.
    _FLY_KEYWORDS = {
        # Explicit schema/organism requests
        "fly schema", "drosophila schema", "using fly", "use fly",
        "fly gene", "fly organism", "fly specific",
        # Species names
        "drosophila", "dmel", "melanogaster", "d. melanogaster", "fruit fly",
        # FlyBase identifiers
        "fbgn", "fbal", "fbtr", "fbbt", "fbdv", "fbrf",
        # Fly-exclusive anatomy (absent in human schemas)
        "wing disc", "eye disc", "leg disc", "imaginal disc",
        "fat body", "malpighian tubule", "polytene", "salivary gland polytene",
        "hemocyte", "ommatidia", "ommatidium",
    }

    def _detect_organism(self, query) -> str:
        query_lower = query.lower()
        if any(kw in query_lower for kw in self._FLY_KEYWORDS):
            return "fly"
        try:
            prompt = ORGANISM_DETECTION_PROMPT.format(query=query)
            result = self.llm.generate(prompt)
            organism = str(result).strip().lower()
            if "fly" in organism:
                return "fly"
            return "human"
        except Exception as e:
            logger.warning(f"Organism detection failed, defaulting to human: {e}")
            return "human"

    def _extract_relevant_information(self, query, enhanced_schema=None):
        try:
            logger.info("Extracting relevant information from the query.")
            schema = enhanced_schema if enhanced_schema is not None else self.enhanced_schema
            prompt = EXTRACT_RELEVANT_INFORMATION_PROMPT.format(
                schema=schema, query=query
            )
            extracted_info = self.llm.generate(prompt)
            logger.info(f"Extracted data: \n{extracted_info}")
            return extracted_info
        except Exception as e:
            logger.error(f"Failed to extract relevant information: {e}")
            raise

    def _convert_to_annotation_json(self, relevant_information, query, enhanced_schema=None):
        try:
            logger.info("Converting relevant information to annotation JSON format.")
            schema = enhanced_schema if enhanced_schema is not None else self.enhanced_schema
            prompt = JSON_CONVERSION_PROMPT.format(
                query=query,
                extracted_information=relevant_information,
                schema=schema,
            )
            json_data = self.llm.generate(prompt)
            logger.info(f"Converted JSON:\n{json.dumps(json_data, indent=2)}")
            return json_data
        except Exception as e:
            logger.error(f"Failed to convert information to annotation JSON: {e}")
            raise

    def _build_lookup_needed(self, updated_json):
        lookup_needed = {}
        for node in updated_json.get("nodes", []):
            node_type = node.get("type")
            node_db_id = node.get("id", "")
            if node_db_id:
                id_prop = self._node_id_property.get(node_type.lower())
                if id_prop:
                    lookup_needed.setdefault((node_type, id_prop), set()).add(node_db_id)
            for property_key, property_value in node.get("properties", {}).items():
                if not property_value and property_value != 0:
                    continue
                if node.get("is_list") and isinstance(property_value, (str, list)):
                    items = (
                        [i.strip() for i in property_value.split(",") if i.strip()]
                        if isinstance(property_value, str) else property_value
                    )
                    lookup_needed.setdefault((node_type, property_key), set()).update(items)
                elif isinstance(property_value, str):
                    lookup_needed.setdefault((node_type, property_key), set()).add(property_value)
        return lookup_needed

    def _build_similarity_cache(self, neo4j, lookup_needed):
        similarity_cache = {}
        for (node_type, property_key), values in lookup_needed.items():
            batch = neo4j.get_similar_property_values_batch(node_type, property_key, list(values))
            for value, matches in batch.items():
                similarity_cache[(node_type, property_key, value)] = matches
        return similarity_cache

    def _build_batch_for_llm(self, updated_json, similarity_cache):
        batch_for_llm = {}
        for node in updated_json.get("nodes", []):
            node_type = node.get("type")
            node_db_id = node.get("id", "")
            if node_db_id:
                id_prop = self._node_id_property.get(node_type.lower())
                if id_prop:
                    similar = similarity_cache.get((node_type, id_prop, node_db_id), [])
                    if similar and similar[0][0].lower() != node_db_id.lower():
                        batch_for_llm[node_db_id] = similar
                    elif not similar:
                        batch_for_llm[node_db_id] = []
            for property_key, property_value in node.get("properties", {}).items():
                if not property_value and property_value != 0:
                    continue
                if node.get("is_list") and isinstance(property_value, (str, list)):
                    items = (
                        [i.strip() for i in property_value.split(",") if i.strip()]
                        if isinstance(property_value, str) else property_value
                    )
                    for item in items:
                        similar = similarity_cache.get((node_type, property_key, item), [])
                        if similar and similar[0][0].lower() != item.lower():
                            batch_for_llm[item] = similar
                        elif not similar:
                            batch_for_llm[item] = []
                elif isinstance(property_value, str):
                    similar = similarity_cache.get((node_type, property_key, property_value), [])
                    if similar and similar[0][0].lower() != property_value.lower():
                        batch_for_llm[property_value] = similar
                    elif not similar:
                        batch_for_llm[property_value] = []
        return batch_for_llm

    def _validate_node_id_field(self, node, node_type, similarity_cache, llm_picks, validation_report):
        node_db_id = node.get("id", "")
        if not node_db_id:
            return
        id_prop = self._node_id_property.get(node_type.lower())
        if not id_prop:
            return
        node_id = node.get("node_id")
        similar_values = similarity_cache.get((node_type, id_prop, node_db_id), [])
        if similar_values and similar_values[0][0].lower() == node_db_id.lower():
            return
        pick = llm_picks.get(node_db_id)
        top = similar_values[0][0] if similar_values else None
        suggestion = pick["value"] if pick else top
        if pick and pick.get("auto_accept"):
            node["id"] = pick["value"]
        else:
            node["status"] = False
            node["needs_confirmation"] = True
            node["pending_substitutions"] = {node_db_id: suggestion or node_db_id}
            validation_report["failed_nodes"].append(
                {"node_id": node_id, "reason": f"'{node_db_id}' not found in database"}
            )

    def _validate_list_property(self, node, node_id, node_type, property_key, property_value,
                                 similarity_cache, llm_picks, validation_report, properties):
        items = (
            [i.strip() for i in property_value.split(",") if i.strip()]
            if isinstance(property_value, str) else property_value
        )
        validated_items, failed_items, item_suggestions = [], [], {}
        for item in items:
            similar_values = similarity_cache.get((node_type, property_key, item), [])
            if not similar_values:
                failed_items.append(item)
                continue
            top = similar_values[0][0]
            if top.lower() == item.lower():
                validated_items.append(top)
                continue
            pick = llm_picks.get(item)
            if pick is None:
                failed_items.append(item)
                item_suggestions[item] = similar_values[0][0]
            elif pick.get("auto_accept"):
                validated_items.append(pick["value"])
            else:
                failed_items.append(item)
                item_suggestions[item] = pick["value"]
        confirmable = {item: item_suggestions[item] for item in failed_items if item in item_suggestions}
        truly_missing = [item for item in failed_items if item not in item_suggestions]
        properties[property_key] = ", ".join(validated_items + truly_missing)
        if failed_items:
            node["status"] = False
            if confirmable:
                node["needs_confirmation"] = True
                node["pending_substitutions"] = confirmable
                node["all_list_values"] = list(items)
            if truly_missing:
                node["not_validated"] = truly_missing
            validation_report["failed_nodes"].append(
                {"node_id": node_id, "reason": f"Could not find in database: {failed_items}"}
            )
        else:
            node["status"] = True

    def _validate_string_property(self, node, node_id, node_type, property_key, property_value,
                                   similarity_cache, llm_picks, validation_report, properties):
        similar_values = similarity_cache.get((node_type, property_key, property_value), [])
        if not similar_values:
            node["status"] = False
            node["validation_error"] = f"'{property_value}' not found in the database."
            validation_report["failed_nodes"].append(
                {"node_id": node_id, "reason": node["validation_error"]}
            )
            return
        top = similar_values[0][0]
        if top.lower() == property_value.lower():
            properties[property_key] = top
            return
        pick = llm_picks.get(property_value)
        if pick is None:
            node["status"] = False
            node["needs_confirmation"] = True
            node["pending_substitutions"] = {property_value: top}
            validation_report["failed_nodes"].append(
                {"node_id": node_id, "reason": f"'{property_value}' not found; nearest is '{top}'"}
            )
        elif pick.get("auto_accept"):
            properties[property_key] = pick["value"]
        else:
            node["status"] = False
            node["needs_confirmation"] = True
            node["pending_substitutions"] = {property_value: pick["value"]}
            validation_report["failed_nodes"].append(
                {"node_id": node_id, "reason": f"'{property_value}' not found; nearest match is '{pick['value']}'"}
            )

    def _validate_node_properties(self, node, node_id, node_type, similarity_cache, llm_picks, validation_report):
        properties = node.get("properties", {})
        for property_key in list(properties.keys()):
            property_value = properties[property_key]
            if not property_value and property_value != 0:
                del properties[property_key]
                validation_report["removed_properties"].append(
                    {"node_type": node_type, "node_id": node_id,
                     "property": property_key, "original_value": property_value}
                )
            elif node.get("is_list") and isinstance(property_value, (str, list)):
                self._validate_list_property(node, node_id, node_type, property_key, property_value,
                                             similarity_cache, llm_picks, validation_report, properties)
            elif isinstance(property_value, str):
                self._validate_string_property(node, node_id, node_type, property_key, property_value,
                                               similarity_cache, llm_picks, validation_report, properties)

    def _validate_edges(self, updated_json, node_types, schema_handler, validation_report):
        valid_predicates = []
        for edge in updated_json.get("predicates", []):
            s = node_types.get(edge["source"])
            t = node_types.get(edge["target"])
            rel = edge["type"]
            conn = f"{s}-{rel}-{t}"
            if conn in schema_handler.processed_schema:
                valid_predicates.append(edge)
            elif f"{t}-{rel}-{s}" in schema_handler.processed_schema:
                validation_report["direction_changes"].append(
                    {"relation_type": rel, "original": f"{s} → {t}", "corrected": f"{t} → {s}"}
                )
                edge["source"], edge["target"] = edge["target"], edge["source"]
                valid_predicates.append(edge)
            else:
                logger.warning(f"Dropping invalid predicate: {conn}")
                validation_report.setdefault("removed_predicates", []).append(
                    {"type": rel, "source": s, "target": t}
                )
        updated_json["predicates"] = valid_predicates

    def _validate_and_update(self, initial_json, neo4j=None, schema_handler=None):
        validation_report = {
            "property_changes": [],
            "direction_changes": [],
            "removed_properties": [],
            "failed_nodes": [],
            "validation_status": "success",
        }
        try:
            logger.info("Validating and updating the JSON structure.")
            _neo4j = neo4j if neo4j is not None else self.neo4j
            _schema_handler = schema_handler if schema_handler is not None else self.schema_handler
            updated_json = copy.deepcopy(initial_json)
            if "nodes" not in updated_json:
                raise ValueError("The input JSON must contain a 'nodes' key.")

            lookup_needed = self._build_lookup_needed(updated_json)
            similarity_cache = self._build_similarity_cache(_neo4j, lookup_needed)
            batch_for_llm = self._build_batch_for_llm(updated_json, similarity_cache)
            llm_picks = self._select_best_matching_values_batch(batch_for_llm)

            node_types = {}
            for node in updated_json.get("nodes", []):
                node_type = node.get("type")
                node_id = node.get("node_id")
                node_types[node_id] = node_type
                if not node.get("is_list"):
                    node["status"] = True
                self._validate_node_id_field(node, node_type, similarity_cache, llm_picks, validation_report)
                self._validate_node_properties(node, node_id, node_type, similarity_cache, llm_picks, validation_report)

            self._validate_edges(updated_json, node_types, _schema_handler, validation_report)

            for node in updated_json.get("nodes", []):
                node.pop("is_list", None)
            updated_json["nodes"] = self._deduplicate_nodes(
                updated_json.get("nodes", []), updated_json.get("predicates", [])
            )
            logger.info(f"Validated and updated JSON: \n{json.dumps(updated_json, indent=2)}")
            return {
                "updated_json": updated_json,
                "validation_report": validation_report,
                "candidates": batch_for_llm,
            }
        except Exception as e:
            logger.error(f"Validation and update of JSON failed: {e}")
            validation_report["validation_status"] = "failed"
            validation_report["error_message"] = str(e)
            return {
                "updated_json": initial_json,
                "validation_report": validation_report,
                "candidates": {},
            }

    def _select_best_matching_property_value(self, user_input_value, possible_values):
        try:
            prompt = SELECT_PROPERTY_VALUE_PROMPT.format(
                search_query=user_input_value, possible_values=possible_values
            )
            selected_value = self.llm.generate(prompt)
            logger.info(f"Selected value: {selected_value}")
            return selected_value
        except Exception as e:
            logger.error(f"Failed to select property value: {e}")
            raise

    def _select_best_matching_values_batch(self, items_with_candidates: dict) -> dict:
        """One LLM call for all ambiguous items.
        Returns {item: {"value": str, "auto_accept": bool} | None}
        """
        if not items_with_candidates:
            return {}
        items_repr = {
            item: [{"value": v, "similarity": round(s, 2)} for v, s in candidates[:5]]
            for item, candidates in items_with_candidates.items()
        }
        prompt = SELECT_PROPERTY_VALUES_BATCH_PROMPT.format(
            items_json=json.dumps(items_repr, indent=2)
        )
        result = self.llm.generate(prompt)
        logger.info(f"Batch LLM picks: {result}")
        if isinstance(result, dict):
            out = {}
            for k, v in result.items():
                if v is None or str(v).lower() == "null":
                    out[k] = None
                elif isinstance(v, dict) and "value" in v:
                    out[k] = {"value": v["value"], "auto_accept": bool(v.get("auto_accept", False))}
                else:
                    out[k] = None
            return out
        return {item: None for item in items_with_candidates}

    # ── Pending state helpers (Redis-backed, in-memory fallback) ─────────────

    def _set_pending(self, user_id: str, data: dict):
        key = f"annotation_pending:{user_id}"
        if self._redis.is_available:
            self._redis.client.set(key, json.dumps(data), ex=self._PENDING_TTL)
        else:
            self._pending_fallback[user_id] = data

    def _get_pending(self, user_id: str):
        key = f"annotation_pending:{user_id}"
        if self._redis.is_available:
            raw = self._redis.client.get(key)
            return json.loads(raw) if raw else None
        return self._pending_fallback.get(user_id)

    def _clear_pending(self, user_id: str):
        key = f"annotation_pending:{user_id}"
        if self._redis.is_available:
            self._redis.client.delete(key)
        else:
            self._pending_fallback.pop(user_id, None)

    # ── Confirmation flow ─────────────────────────────────────────────────────

    def has_pending_for(self, user_id: str) -> bool:
        return self._get_pending(user_id) is not None

    def handle_confirmation_response(self, user_id: str, query: str):
        """Call from assistant_response when a pending confirmation exists.
        Returns a ready response dict, or None if query is a new unrelated question.
        """
        entry = self._get_pending(user_id)
        if not entry:
            return None

        pending_json = entry["json"]
        candidates  = entry.get("candidates", {})
        unconfirmed = entry.get("unconfirmed", [])
        pending_organism = entry.get("organism", "human")

        verdict = self._classify_confirmation(query)

        if verdict == "confirm":
            resolved = self._apply_pending_substitutions(pending_json, apply=True)
            self._clear_pending(user_id)
            return {
                "text": "Got it! I've built the annotation structure using the confirmed match. The structured data is ready.",
                "json_format": resolved,
                "organism": pending_organism,
                "agents_completed": ["annotation_agent"],
            }

        if verdict == "reject":
            resolved = self._apply_pending_substitutions(pending_json, apply=False)
            self._clear_pending(user_id)
            return {
                "text": "Understood! I've built the annotation structure without the unidentified node. The structured data is ready.",
                "json_format": resolved,
                "organism": pending_organism,
                "agents_completed": ["annotation_agent"],
            }

        if verdict == "show_alternatives":
            # Show the other Neo4j candidates — keep pending active so user can still confirm/reject
            lines = []
            for u in unconfirmed:
                original  = u["original"]
                all_cands = candidates.get(original, [])
                # Skip the already-suggested top hit; show the rest
                suggestion = u["suggestion"]
                others = [(v, s) for v, s in all_cands if v != suggestion]
                if others:
                    others_str = ", ".join(f"**{v}** ({round(s*100)}% similar)" for v, s in others[:4])
                    lines.append(
                        f"Other candidates for **'{original}'** in the database: {others_str}.\n"
                        f"The closest remains **'{suggestion}'**. "
                        f"Reply with the name you'd like to use, or say **yes** to use '{suggestion}', "
                        f"or **no** to build without it."
                    )
                else:
                    lines.append(
                        f"There are no other similar entries for **'{original}'** in the database. "
                        f"The only close match is **'{suggestion}'**. "
                        f"Say **yes** to use it or **no** to build without it."
                    )
            return {
                "text": "\n\n".join(lines),
                "json_format": None,
                "agents_completed": ["annotation_agent"],
            }

        # New unrelated query — clear stale state and proceed normally
        self._clear_pending(user_id)
        return None

    def _classify_confirmation(self, message: str) -> str:
        """Uses the LLM to understand what the user meant — no keyword matching."""
        prompt = (
            f"The assistant asked the user to confirm whether to substitute an unrecognised "
            f"database entry with a suggested match. The user replied:\n\n"
            f"\"{message}\"\n\n"
            f"Classify the user's intent as exactly one of:\n"
            f"- confirm           — user agrees to use the suggested match "
            f"(e.g. 'yes', 'sure', 'use it', 'go ahead', 'that works', 'use ZNF697')\n"
            f"- reject            — user wants to build without the unidentified node "
            f"(e.g. 'no', 'skip it', 'build without', 'leave it out')\n"
            f"- show_alternatives — user wants to see other possible matches from the database "
            f"(e.g. 'find another', 'show me others', 'what else is there', 'forgot the name')\n"
            f"- new_query         — user is asking something completely unrelated to the confirmation\n\n"
            f"Reply with only one word: confirm, reject, show_alternatives, or new_query."
        )
        try:
            result = self.llm.generate(prompt)
            verdict = result.strip().lower().split()[0] if result else "new_query"
            return verdict if verdict in ("confirm", "reject", "show_alternatives", "new_query") else "new_query"
        except Exception:
            return "new_query"

    def _apply_id_substitution(self, node, subs):
        node_id_val = node.get("id", "")
        if not node_id_val or node_id_val not in subs:
            return
        suggested = subs[node_id_val]
        id_prop = self._node_id_property.get(node.get("type", "").lower())
        if id_prop:
            node.setdefault("properties", {})[id_prop] = suggested
        node["id"] = ""

    def _apply_property_substitutions_to_node(self, node, subs):
        for prop_key, prop_val in node.get("properties", {}).items():
            if not isinstance(prop_val, str):
                continue
            parts = [p.strip() for p in prop_val.split(",") if p.strip()]
            replaced = set()
            new_parts = []
            for part in parts:
                replacement = next(
                    (sugg for orig, sugg in subs.items() if part.lower() == orig.lower()),
                    None,
                )
                if replacement:
                    new_parts.append(replacement)
                    replaced.add(part.lower())
                else:
                    new_parts.append(part)
            existing_lower = {p.lower() for p in new_parts}
            for orig, sugg in subs.items():
                if orig.lower() not in replaced and sugg.lower() not in existing_lower:
                    new_parts.append(sugg)
                    existing_lower.add(sugg.lower())
            node["properties"][prop_key] = ", ".join(new_parts)

    def _apply_pending_substitutions(self, pending_json: dict, apply: bool = True) -> dict:
        result = copy.deepcopy(pending_json)
        for node in result.get("nodes", []):
            if not (node.get("needs_confirmation") and node.get("pending_substitutions")):
                continue
            if apply:
                subs = node["pending_substitutions"]
                self._apply_id_substitution(node, subs)
                self._apply_property_substitutions_to_node(node, subs)
            node["status"] = True
            for key in ("needs_confirmation", "pending_substitutions", "all_list_values",
                        "not_validated", "validation_error"):
                node.pop(key, None)
        result["nodes"] = self._deduplicate_nodes(result.get("nodes", []), result.get("predicates", []))
        return result

    def _deduplicate_nodes(self, nodes: list, predicates: list) -> list:
        """Remove nodes whose type+properties are exact duplicates of an earlier node.
        Predicates that reference a removed duplicate are remapped to the surviving node.
        """
        seen = {}       # (type, frozenset(properties.items())) -> surviving node_id
        removed = {}    # removed node_id -> surviving node_id
        kept = []
        for node in nodes:
            key = (node.get("type", ""), frozenset(
                (k, v) for k, v in node.get("properties", {}).items()
            ))
            if key in seen:
                removed[node["node_id"]] = seen[key]
            else:
                seen[key] = node["node_id"]
                kept.append(node)

        # Remap predicates
        if removed:
            for pred in predicates:
                if pred.get("source") in removed:
                    pred["source"] = removed[pred["source"]]
                if pred.get("target") in removed:
                    pred["target"] = removed[pred["target"]]

        return kept

    def _describe_annotation_result(self, query: str, validated_json: dict) -> str:
        """Generate a meaningful biological description from the query + validated JSON.
        Replaces the generic 'structure created successfully' text so the aggregator
        has real content to work with instead of hallucinating.
        """
        try:
            nodes = validated_json.get("nodes", [])
            predicates = validated_json.get("predicates", [])

            node_id_to_label = {}
            node_lines = []
            for n in nodes:
                nid  = n.get("node_id", "")
                ntype = n.get("type", "unknown")
                props = n.get("properties", {})
                prop_str = ", ".join(f"{k}: {v}" for k, v in props.items()) if props else "(no properties)"
                node_lines.append(f"- {ntype} [{nid}]: {prop_str}")
                node_id_to_label[nid] = f"{ntype}({prop_str})"

            pred_lines = []
            for p in predicates:
                src = node_id_to_label.get(p.get("source", ""), p.get("source", ""))
                tgt = node_id_to_label.get(p.get("target", ""), p.get("target", ""))
                pred_lines.append(f"- {src} --[{p.get('type', '')}]--> {tgt}")

            structure_summary = "Nodes:\n" + "\n".join(node_lines)
            if pred_lines:
                structure_summary += "\n\nRelationships:\n" + "\n".join(pred_lines)

            prompt = (
                f"A user asked: \"{query}\"\n\n"
                f"The following annotation structure was built from the database schema:\n\n"
                f"{structure_summary}\n\n"
                f"Write 1-3 sentences describing what this structure represents biologically "
                f"and what query it will run. Be specific about the entities and relationships "
                f"involved. Do NOT invent data, relationships, or biological facts not shown above. "
                f"Do NOT mention the annotation system or technical details."
            )
            result = self.llm.generate(prompt)
            return result.strip() if result else "The annotation structure was built successfully."
        except Exception as e:
            logger.warning(f"Failed to generate annotation description: {e}")
            return "The annotation structure was built successfully."

    def _build_confirmation_text(self, unconfirmed_nodes: list) -> str:
        if len(unconfirmed_nodes) == 1:
            u = unconfirmed_nodes[0]
            all_vals = u.get("all_list_values") or []
            known_vals = [v for v in all_vals if v != u["original"]]

            base = (
                f"I couldn't find **'{u['original']}'** in the database. "
                f"The closest match I found is **'{u['suggestion']}'**.\n\n"
                f"Should I go ahead and use **'{u['suggestion']}'** in place of **'{u['original']}'**?"
            )
            if known_vals:
                base += f" Or would you like me to build the annotation without it, using only {known_vals}?"
            else:
                base += " Or would you like to cancel and try a different identifier?"
            return base

        lines = ["I couldn't find some of the nodes you mentioned in the database:"]
        for u in unconfirmed_nodes:
            lines.append(
                f"  - **'{u['original']}'** — closest match is **'{u['suggestion']}'**"
            )
        lines.append(
            "\nShould I go ahead with these substitutions? "
            "Or would you prefer I build the annotation skipping the unrecognised nodes?"
        )
        return "\n".join(lines)

    def _process_query_record(self, record, nodes, relationships, node_ids, rel_ids, data, records):
        record_data = {}
        for key, value in record.items():
            if hasattr(value, "labels"):
                node_data = {"id": str(value.id), "labels": list(value.labels), "properties": dict(value)}
                if str(value.id) not in node_ids:
                    nodes.append(node_data)
                    node_ids.add(str(value.id))
            elif hasattr(value, "type"):
                rel_data = {
                    "id": str(value.id), "type": value.type,
                    "start_node": str(value.start_node.id), "end_node": str(value.end_node.id),
                    "properties": dict(value),
                }
                if str(value.id) not in rel_ids:
                    relationships.append(rel_data)
                    rel_ids.add(str(value.id))
            else:
                data[key] = value
                record_data[key] = value
        if record_data:
            records.append(record_data)

    def execute_cypher_query(self, cypher_query):
        try:
            logger.info(f"Executing Cypher query: {cypher_query}")
            driver = self.neo4j.get_driver()
            with driver.session() as session:
                result = session.run(cypher_query)
                nodes, relationships, records = [], [], []
                node_ids, rel_ids = set(), set()
                data = {}
                for record in result:
                    self._process_query_record(record, nodes, relationships, node_ids, rel_ids, data, records)
                counts = {
                    "total_nodes": len(nodes),
                    "total_relationships": len(relationships),
                    "result_records": len(list(result.data())),
                }
                is_path_query = any(
                    rel in cypher_query.lower()
                    for rel in ["transcribes_to", "part_of", "transcribed_from", "translates_to"]
                )
                if is_path_query and counts["total_relationships"] == 0:
                    logger.warning(f"Path query returned no relationships: {cypher_query}")
                logger.info(f"Query executed successfully. Found {counts['total_nodes']} nodes and {counts['total_relationships']} relationships")
                return {
                    "success": True,
                    "data": {"nodes": nodes, "relationships": relationships, "counts": counts, "records": records, **data},
                    "error": None,
                    "cypher_query": cypher_query,
                }
        except Exception as e:
            error_msg = f"Error executing Cypher query: {str(e)}"
            logger.error(error_msg)
            return {
                "success": False,
                "data": {"nodes": [], "relationships": [], "counts": {"total_nodes": 0, "total_relationships": 0, "result_records": 0}},
                "error": error_msg,
                "cypher_query": cypher_query,
            }

    def _build_summary_data(self, query, nodes, relationships):
        summary_data = {
            "original_query": query,
            "nodes_found": len(nodes),
            "relationships_found": len(relationships),
            "node_types": {},
            "key_properties": {},
            "relationships_info": [],
        }
        key_props = {"gene_name", "transcript_id", "exon_id", "gene_type", "chr"}
        for node in nodes:
            node_type = node.get("labels", ["unknown"])[0] if node.get("labels") else "unknown"
            summary_data["node_types"].setdefault(node_type, [])
            key_info = {p: v for p, v in node.get("properties", {}).items() if p in key_props}
            if key_info:
                summary_data["node_types"][node_type].append(key_info)
        for rel in relationships:
            summary_data["relationships_info"].append({
                "type": rel.get("type", "unknown"),
                "start_node": rel.get("start_node", "unknown"),
                "end_node": rel.get("end_node", "unknown"),
            })
        return summary_data

    def summarize_results(self, query, results):
        try:
            logger.info(f"Starting result summarization for query: '{query}'")
            if not results.get("success", False):
                error_msg = results.get("error", "Unknown error occurred")
                logger.error(f"Cannot summarize failed query results: {error_msg}")
                return f"I'm sorry, but I encountered an error while searching the database: {error_msg}"
            data = results.get("data", {})
            nodes = data.get("nodes", [])
            relationships = data.get("relationships", [])
            counts = data.get("counts", {})
            if counts.get("total_nodes", 0) == 0:
                return f"I searched for information about '{query}', but I couldn't find any matching data in the database. Please try rephrasing your question or check if the gene/transcript/exon names are correct."
            summary_data = self._build_summary_data(query, nodes, relationships)
            summarization_prompt = self._create_summarization_prompt(query, summary_data)
            logger.info("Generating summary using LLM...")
            summary = self.llm.generate(summarization_prompt)
            logger.info(f"Successfully generated summary: {summary[:100]}...")
            return summary
        except Exception as e:
            logger.error(f"Error during result summarization: {str(e)}")
            try:
                counts = results.get("data", {}).get("counts", {})
                if counts.get("total_nodes", 0) > 0:
                    return f"I found {counts['total_nodes']} items related to your query '{query}'. However, I encountered an issue while generating a detailed summary."
                return f"I couldn't find any information for '{query}' in the database."
            except Exception:
                return f"I'm sorry, but I encountered an error while processing your query '{query}'."

    def _create_summarization_prompt(self, query, summary_data):
        # Create a prompt for the LLM to generate user-friendly summaries.
        # Build node summary
        node_summary = ""
        for node_type, nodes_list in summary_data["node_types"].items():
            node_summary += f"\n- {node_type.capitalize()} nodes: {len(nodes_list)}"
            for node_info in nodes_list[:3]:  # Show first 3 nodes
                node_summary += f"\n  • {node_type.capitalize()}: "
                for prop, value in node_info.items():
                    node_summary += f"{prop}={value}, "
                node_summary = node_summary.rstrip(", ") + "\n"

        # Build relationship summary
        relationship_summary = ""
        if summary_data["relationships_found"] > 0:
            relationship_summary = f"\n**Relationships Found:**\n"
            for rel in summary_data["relationships_info"][
                :5
            ]:  # Show first 5 relationships
                relationship_summary += f"- {rel['type']}: connects nodes {rel['start_node']} and {rel['end_node']}\n"
        else:
            relationship_summary = "No relationships found."

        return RESULT_SUMMARIZATION_PROMPT.format(
            query=query,
            node_summary=node_summary,
            relationship_summary=relationship_summary,
        )

    def process_annotation_query(
        self, query, user_id, query_type="annotation_biological"
    ):
        # orchestrate the entire annotation pipeline from user query to final response
        try:
            logger.info(
                f"Starting annotation pipeline for query: '{query}', type: {query_type}"
            )

            # Route based on query type
            if query_type == "annotation_general":
                return self._handle_general_query(query, user_id)
            else:
                return self._handle_biological_query(query, user_id)

        except Exception as e:
            error_msg = f"Unexpected error in annotation pipeline: {str(e)}"
            logger.error(error_msg)
            return {
                "success": False,
                "error": error_msg,
                "pipeline_status": {
                    "json_extraction": "unknown",
                    "cypher_conversion": "unknown",
                    "database_execution": "unknown",
                    "summarization": "unknown",
                },
            }

    def _handle_biological_query(self, query, user_id):
        try:
            # Detect organism and select the right schema + Neo4j resources
            organism = self._detect_organism(query)
            if organism == "fly" and self.fly_schema_handler and self.fly_neo4j:
                active_schema = self.fly_enhanced_schema
                active_schema_handler = self.fly_schema_handler
                active_neo4j = self.fly_neo4j
                logger.info("Organism detected: fly — using fly schema and Neo4j")
            else:
                organism = "human"
                active_schema = self.enhanced_schema
                active_schema_handler = self.schema_handler
                active_neo4j = self.neo4j
                logger.info("Organism detected: human — using human schema and Neo4j")

            # Extract and validate JSON query
            emit_to_user(
                user=user_id,
                message="Extracting relevant information from your query...",
            )
            try:
                relevant_information = self._extract_relevant_information(query, enhanced_schema=active_schema)
                logger.info("Relevant information extraction successful")

                # Convert to initial JSON
                emit_to_user(
                    user=user_id, message="Validating Constructed Json Format..."
                )
                initial_json = self._convert_to_annotation_json(
                    relevant_information, query, enhanced_schema=active_schema
                )
                logger.info("Initial JSON conversion successful")

                # Validate and update
                validation = self._validate_and_update(initial_json, neo4j=active_neo4j, schema_handler=active_schema_handler)
                logger.info("JSON validation successful")

                # Collect nodes that need user confirmation before the JSON can be finalised
                unconfirmed_nodes = []
                for node in validation["updated_json"].get("nodes", []):
                    if node.get("needs_confirmation") and node.get("pending_substitutions"):
                        for original, suggestion in node["pending_substitutions"].items():
                            unconfirmed_nodes.append({
                                "node_id": node.get("node_id"),
                                "node_type": node.get("type"),
                                "original": original,
                                "suggestion": suggestion,
                                "all_list_values": node.get("all_list_values", []),
                            })

                if unconfirmed_nodes:
                    logger.info(f"Returning needs_confirmation for {len(unconfirmed_nodes)} node(s)")
                    self._set_pending(user_id, {
                        "json": validation["updated_json"],
                        "candidates": validation.get("candidates", {}),
                        "unconfirmed": unconfirmed_nodes,
                        "organism": organism,
                    })
                    return {
                        "success": True,
                        "needs_confirmation": True,
                        "confirmation_text": self._build_confirmation_text(unconfirmed_nodes),
                        "summary": None,
                        "json_format": None,
                        "validation_report": validation["validation_report"],
                        "resource": {"id": None, "type": "annotation"},
                    }

                summary = self._describe_annotation_result(query, validation["updated_json"])

                updated_json = validation["updated_json"]
                json_query = {
                    "success": True,
                    "summary": summary,
                    "json_format": updated_json,
                    "organism": organism,
                    "validation_report": validation["validation_report"],
                    "resource": {"id": None, "type": "annotation"},
                }

                logger.info("JSON query extraction successful")
                logger.info(f"JSON query structure: {json.dumps(json_query, indent=2)}")

                return json_query
            
            except Exception as e:
                logger.error(f"Failed to extract JSON query: {str(e)}")
                return {
                    "success": False,
                    "error": f"Failed to process query: {str(e)}",
                    "pipeline_status": {"json_extraction": "failed"},
                }

        except Exception as e:
            error_msg = f"Unexpected error in biological query pipeline: {str(e)}"
            logger.error(error_msg)
            return {
                "success": False,
                "error": error_msg,
                "pipeline_status": {
                    "json_extraction": "unknown",
                    "cypher_conversion": "unknown",
                    "database_execution": "unknown",
                    "summarization": "unknown",
                },
            }

    def _handle_general_query(self, query, user_id):
        try:
            logger.info(f"Handling general query: '{query}'")

            emit_to_user(
                user=user_id,
                message="Analyzing database information...",
            )

            # Generate simple database summary
            database_summary = self._generate_database_summary()

            # Use LLM to answer the query based on the summary
            summary_prompt = f"""
            Based on this database summary: {database_summary}
            
            Answer this question: {query}
            
            Provide a clear, informative response based on the available data.
            """

            summary = self.llm.generate(summary_prompt)
            logger.info("General query answered successfully")

            return {
                "success": True,
                "summary": summary,
                "cypher_query": None,
                "json_query": None,
                "database_results": {"data": {"summary": database_summary}},
                "error": None,
                "pipeline_status": {
                    "general_query_handling": "success",
                },
            }

        except Exception as e:
            error_msg = f"Error handling general query: {str(e)}"
            logger.error(error_msg)
            return {
                "success": False,
                "error": error_msg,
                "pipeline_status": {
                    "general_query_handling": "failed",
                },
            }

    def _format_stats_record(self, key, data, records):
        value = data.get(key)
        if value is not None:
            return f"{key}: {value}"
        if not records:
            return f"{key}: No data found"
        if key == "node_types":
            parts = [f"{r.get('node_type', 'unknown')} ({r.get('count', 0)})" for r in records]
            return f"{key}: {', '.join(parts)}"
        if key == "relationship_types":
            parts = [f"{r.get('rel_type', 'unknown')} ({r.get('count', 0)})" for r in records]
            return f"{key}: {', '.join(parts)}"
        return f"{key}: {records}"

    def _generate_database_summary(self):
        try:
            stats_queries = {
                "total_nodes": "MATCH (n) RETURN count(n) as total_nodes",
                "total_relationships": "MATCH ()-[r]->() RETURN count(r) as total_relationships",
                "node_types": "MATCH (n) RETURN DISTINCT labels(n)[0] as node_type, count(n) as count ORDER BY count DESC",
                "relationship_types": "MATCH ()-[r]->() RETURN DISTINCT type(r) as rel_type, count(r) as count ORDER BY count DESC",
            }
            summary_parts = []
            for key, query in stats_queries.items():
                try:
                    result = self.execute_cypher_query(query)
                    if result.get("success"):
                        data = result.get("data", {})
                        summary_parts.append(self._format_stats_record(key, data, data.get("records", [])))
                    else:
                        summary_parts.append(f"{key}: Unable to retrieve")
                except Exception as e:
                    logger.warning(f"Failed to execute {key} query: {e}")
                    summary_parts.append(f"{key}: Error retrieving")
            return "Database Summary:\n" + "\n".join(summary_parts)
        except Exception as e:
            logger.error(f"Failed to generate database summary: {e}")
            return "Database Summary:\nUnable to retrieve database information due to an error."
