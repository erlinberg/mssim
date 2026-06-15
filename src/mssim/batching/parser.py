import json
import logging
from importlib.resources import files
from typing import Any

from jsonschema import validate


logger = logging.getLogger(__name__)

JSON_SCHEMA_FILENAME = "settings_schema.json"


class MSSIMConfigVariable:
    def __init__(
        self,
        name: str,
        json_path: str,
        json_key: str,
        default_value: Any = None,
        is_kwarg: bool = False,
        conditions: dict[str, list[Any]] | None = None,
    ) -> None:
        self.name = name
        self.json_path = json_path
        self.json_key = json_key
        self.value = default_value

        self.is_kwarg = is_kwarg
        self.conditions = conditions or {}

    @property
    def cardinality(self) -> int:
        """Returns the number of configurations this variable contributes to when sweeping."""
        return len(self.value) if isinstance(self.value, list) else 0

    def is_active_under(self, parent_state: dict[str, Any]) -> bool:
        """Evaluates if this variable should be active given the current parent configuration."""
        if not self.conditions:
            return True

        for condition_key, allowed_values in self.conditions.items():
            if parent_state.get(condition_key) not in allowed_values:
                return False

        return True


class MSSIMParser:
    def __init__(self, config_path: str) -> None:
        self.config_path = config_path
        self.data = self.load_config()

        self.validate_schema()

        # Define all possible variables with their JSON paths, keys, and conditions
        self.variables = [
            MSSIMConfigVariable(
                name="n_qubits", json_path="model", json_key="n_qubits"
            ),
            MSSIMConfigVariable(name="depth", json_path="model", json_key="depth"),
            MSSIMConfigVariable(
                name="observable", json_path="model", json_key="observable"
            ),
            MSSIMConfigVariable(name="verbose", json_path="output", json_key="verbose"),
            MSSIMConfigVariable(
                name="engine", json_path="execution", json_key="engine"
            ),
            MSSIMConfigVariable(name="circuit", json_path="model", json_key="circuit"),
            MSSIMConfigVariable(
                name="n_runs", json_path="execution", json_key="n_runs"
            ),
            MSSIMConfigVariable(
                name="max_bond_dimension",
                json_path="execution",
                json_key="max_bond_dimension",
                conditions={"engine": ["quimb", "mpstab", "all"]},
            ),
            MSSIMConfigVariable(
                name="max_terms",
                json_path="execution",
                json_key="max_terms",
                conditions={"engine": ["qiskit_paulipropagation", "all"]},
            ),
            MSSIMConfigVariable(
                name="J",
                json_path="model.kwargs",
                json_key="J",
                is_kwarg=True,
                conditions={"circuit": ["kicked-ising", "ising"]},
            ),
            MSSIMConfigVariable(
                name="h",
                json_path="model.kwargs",
                json_key="h",
                is_kwarg=True,
                conditions={"circuit": ["kicked-ising", "ising"]},
            ),
            MSSIMConfigVariable(
                name="dt",
                json_path="model.kwargs",
                json_key="dt",
                is_kwarg=True,
                conditions={"circuit": ["ising"]},
            ),
            MSSIMConfigVariable(
                name="b",
                json_path="model.kwargs",
                json_key="b",
                is_kwarg=True,
                conditions={"circuit": ["kicked-ising"]},
            ),
            MSSIMConfigVariable(
                name="K",
                json_path="model.kwargs",
                json_key="K",
                is_kwarg=True,
                conditions={"circuit": ["cme"]},
            ),
            MSSIMConfigVariable(
                name="M",
                json_path="model.kwargs",
                json_key="M",
                is_kwarg=True,
                conditions={"circuit": ["cme"]},
            ),
        ]

        self.active_variables = []

        self.parse_variables()

    def load_config(self) -> dict[str, Any]:
        """Loads the JSON configuration file and returns it as a dictionary."""
        logger.info("Loading config from %s", self.config_path)
        with open(self.config_path, "r") as f:
            return json.load(f)

    def resolve_json_path(self, path: str) -> dict[str, Any]:
        """Resolves a dot-separated JSON path into a nested dictionary."""
        keys = path.split(".")
        current = self.data

        for key in keys:
            if not isinstance(current, dict) or key not in current:
                return {}
            current = current[key]

        return current

    def validate_schema(self) -> None:
        """Validates the loaded configuration against a predefined JSON schema."""
        logger.info("Validating config against schema %s", JSON_SCHEMA_FILENAME)

        schema_text = files("mssim").joinpath(JSON_SCHEMA_FILENAME).read_text()
        schema = json.loads(schema_text)

        # Soft validation: log warnings for unknown fields instead of raising errors
        try:
            validate(instance=self.data, schema=schema)
            logger.info("Config validation successful")
        except Exception as e:
            logger.warning("Config validation failed: %s", e)

    def parse_variables(self) -> None:
        """Parses the variables from the config file, handling both sweeping and static variables."""
        sweep_block = self.data.get("sweep")

        for var in self.variables:
            if sweep_block:
                # Logic to handle sweeping variables (e.g., generating combinations)
                var.value = sweep_block.get(var.name)

                if var.value is not None:
                    if not isinstance(var.value, list):
                        var.value = [var.value]  # Wrap single values in a list
                    self.active_variables.append(var)
                    continue  # If the variable is found in the sweep block, skip to the next variable

            loc = self.resolve_json_path(var.json_path)

            var.value = loc.get(var.name)
            if var.value is not None:
                if not isinstance(var.value, list):
                    var.value = [var.value]  # Wrap single values in a list
                self.active_variables.append(var)

        logger.debug(
            "Parsed %d active variables (%d kwargs)",
            len(self.active_variables),
            sum(1 for var in self.active_variables if var.is_kwarg),
        )

    def _combine_kwargs(self, sim_config: dict[str, Any]) -> dict[str, Any]:
        """Combine kwarg variables into a nested kwargs dictionary."""
        combined_config = sim_config.copy()
        kwarg_names = {var.name for var in self.active_variables if var.is_kwarg}

        kwargs = {
            key: combined_config.pop(key)
            for key in kwarg_names
            if key in combined_config
        }

        if kwargs:
            combined_config["kwargs"] = kwargs

        return combined_config
    
    def post_process(self, raw_config: dict[str, Any], task_id) -> dict[str, Any]:
        """Apply any necessary post-processing to the config before running the simulation."""
        # For now, this just combines kwargs, but additional logic can be added here if needed.
        base_config = self._combine_kwargs(raw_config)
        output_settings = self.data.get("output", {})

        return {
            **base_config,

            # Output settings
            "filename": output_settings.get("filename", "results.jsonl"),
            "format": output_settings.get("format", "jsonl"),

            # Metadata
            "run_id": task_id,
            "settings_file": self.config_path,
        }
        
