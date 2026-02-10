"""
Feature plugin registry.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Optional

import pandas as pd


FeatureGenerator = Callable[[pd.DataFrame, dict[str, Any], Optional[dict[str, Any]]], pd.DataFrame]


@dataclass
class FeaturePlugin:
    """A single feature generator plugin."""

    name: str
    generator: FeatureGenerator
    group: str
    description: str = ""


@dataclass
class FeatureSpec:
    """Declarative feature generation spec."""

    name: str
    params: dict[str, Any] = field(default_factory=dict)


class FeatureRegistry:
    """
    Registry for composable feature plugins.
    """

    def __init__(self):
        self._plugins: dict[str, FeaturePlugin] = {}

    def register(self, plugin: FeaturePlugin) -> None:
        self._plugins[plugin.name] = plugin

    def unregister(self, name: str) -> None:
        self._plugins.pop(name, None)

    def has(self, name: str) -> bool:
        return name in self._plugins

    def get(self, name: str) -> FeaturePlugin:
        if name not in self._plugins:
            raise KeyError(f"Feature plugin not registered: {name}")
        return self._plugins[name]

    def list_plugins(self) -> list[dict[str, str]]:
        return [
            {
                "name": plugin.name,
                "group": plugin.group,
                "description": plugin.description,
            }
            for plugin in self._plugins.values()
        ]

    def generate(
        self,
        base_df: pd.DataFrame,
        specs: Optional[list[FeatureSpec]] = None,
        context: Optional[dict[str, Any]] = None,
        include_base_columns: bool = True,
    ) -> tuple[pd.DataFrame, dict[str, Any]]:
        """
        Generate features from registered plugins.
        """
        if specs is None:
            specs = [FeatureSpec(name=name) for name in self._plugins]

        if include_base_columns:
            output = base_df.copy()
        else:
            output = pd.DataFrame(index=base_df.index)

        catalog: list[dict[str, Any]] = []
        all_feature_names: list[str] = []

        for spec in specs:
            plugin = self.get(spec.name)
            feature_frame = plugin.generator(base_df, spec.params, context)
            if feature_frame is None or feature_frame.empty:
                continue

            output = pd.concat([output, feature_frame], axis=1)
            feature_names = list(feature_frame.columns)
            all_feature_names.extend(feature_names)
            catalog.append(
                {
                    "plugin": plugin.name,
                    "group": plugin.group,
                    "params": spec.params,
                    "features": feature_names,
                }
            )

        metadata = {
            "feature_names": all_feature_names,
            "catalog": catalog,
            "plugin_count": len(catalog),
        }
        return output, metadata
