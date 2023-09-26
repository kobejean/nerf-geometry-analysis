"""
Nerfstudio Template Config

Define your custom method here that registers with Nerfstudio CLI.
"""

from __future__ import annotations

from nerfstudio.plugins.registry_dataparser import DataParserSpecification
from nga.data.dataparsers.nga_dataparser import NGADataparser

nga_dataparser = DataParserSpecification(config=NGADataparser())
