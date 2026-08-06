import sys
import types
import unittest
from unittest import mock


from cutlery_remote.registry_proxy import (
    REGISTRY_OPERATIONS,
    SHARED_EXTENSION_MODULE,
    RegistryProxyRequestError,
    _install_shared_registry_operations,
    prepare_registry_operation,
)


class RemoteRegistryProxyContractTests(unittest.TestCase):
    def test_registry_ids_map_to_exact_methods_and_paths(self):
        expected = {
            "remote_clip.choices": ("GET", "/cutlery/remote/clip/choices"),
        }

        self.assertEqual(
            {
                registry_id: (operation.method, operation.path)
                for registry_id, operation in REGISTRY_OPERATIONS.items()
            },
            expected,
        )

    def test_unknown_registry_cannot_supply_an_arbitrary_path(self):
        with self.assertRaises(RegistryProxyRequestError) as raised:
            prepare_registry_operation(
                "https://attacker.example/arbitrary",
                {"path": "/admin"},
            )

        self.assertEqual(raised.exception.code, "unknown_registry")

    def test_operation_rejects_payload_fields_outside_its_allowlist(self):
        with self.assertRaises(RegistryProxyRequestError) as raised:
            prepare_registry_operation(
                "remote_clip.choices",
                {
                    "authorization": "Bearer must-not-be-forwarded",
                },
            )

        self.assertEqual(
            raised.exception.code,
            "unsupported_registry_payload_fields",
        )

    def test_remote_registry_surface_is_read_only(self):
        with self.assertRaises(RegistryProxyRequestError) as raised:
            prepare_registry_operation(
                "remote_clip.choices.write",
                {"name": "must-not-write-remotely"},
            )

        self.assertEqual(raised.exception.code, "unknown_registry")

    def test_get_registry_requires_an_empty_payload(self):
        with self.assertRaises(RegistryProxyRequestError) as raised:
            prepare_registry_operation("remote_clip.choices", {"path": "/other"})

        self.assertEqual(
            raised.exception.code,
            "unsupported_registry_payload_fields",
        )

    def test_registry_payload_must_be_an_object(self):
        with self.assertRaises(RegistryProxyRequestError) as raised:
            prepare_registry_operation("remote_clip.choices", ["not", "an", "object"])
        self.assertEqual(raised.exception.code, "invalid_registry_payload")

    def test_private_contracts_can_be_published_before_public_module_loads(self):
        extension = types.ModuleType(SHARED_EXTENSION_MODULE)
        extension.registry_operations = {
            "private.example": ("POST", "/cutlery/private/example", frozenset({"model"})),
            "unsafe.example": ("DELETE", "https://attacker.example", frozenset()),
        }
        with mock.patch.dict(sys.modules, {SHARED_EXTENSION_MODULE: extension}):
            installed = _install_shared_registry_operations()
        try:
            self.assertEqual(installed, {"private.example"})
            self.assertEqual(REGISTRY_OPERATIONS["private.example"].path, "/cutlery/private/example")
            self.assertNotIn("unsafe.example", REGISTRY_OPERATIONS)
        finally:
            REGISTRY_OPERATIONS.pop("private.example", None)


if __name__ == "__main__":
    unittest.main()
