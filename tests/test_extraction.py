import pytest

from legal_chatbot.extraction import (
    ALL_FIELDS,
    ContractField,
    ContractInfo,
    Signatory,
    detect_fields,
    extract_json_object,
    format_answer,
    is_meaningful,
    normalize_date,
)


class TestDetectFields:
    @pytest.mark.parametrize(
        ("question", "expected"),
        [
            ("Quel est le nom du client ?", [ContractField.CLIENT]),
            ("Quel est le type de contrat ?", [ContractField.CONTRACT_TYPE]),
            ("Quelle est la date du contrat ?", [ContractField.CONTRACT_DATE]),
            ("Qui sont les signataires ?", [ContractField.SIGNATORIES]),
            ("Quel est le MONTANT total ?", [ContractField.AMOUNT]),
            ("Who signed the agreement?", [ContractField.SIGNATORIES]),
            ("Quand les signataires ont-ils signé ?", [ContractField.SIGNATURE_DATES]),
            (
                "What is the contract type and the price?",
                [ContractField.CONTRACT_TYPE, ContractField.AMOUNT],
            ),
        ],
    )
    def test_targeted_questions(self, question, expected):
        assert detect_fields(question) == expected

    def test_summary_asks_for_every_field(self):
        assert detect_fields("Donne-moi un résumé du contrat") == list(ALL_FIELDS)

    @pytest.mark.parametrize(
        "question",
        [
            "Que prévoit la clause de confidentialité ?",
            "Can the licensee update the software?",  # "update" must not match "date"
            "Quelle est la loi applicable ?",
        ],
    )
    def test_open_questions_are_not_routed_to_extraction(self, question):
        assert detect_fields(question) == []


class TestIsMeaningful:
    @pytest.mark.parametrize(
        "value", ["ACME Corp", "48 000 EUR", "Confidentiel", ["", "Jane Doe"], 42]
    )
    def test_real_values(self, value):
        assert is_meaningful(value)

    @pytest.mark.parametrize(
        "value",
        [
            None,
            "",
            "   ",
            "Non précisé",
            "N/A",
            "_____",
            "[Name]",
            "Insert legal name",
            "name of signatory",
            [],
        ],
    )
    def test_placeholders(self, value):
        assert not is_meaningful(value)

    def test_regression_any_string_was_rejected(self):
        # The first version listed "" among placeholders and used a substring test,
        # so every string was considered invalid.
        assert is_meaningful("Northwind Analytics SARL")

    def test_word_containing_placeholder_is_kept(self):
        # "date" is a placeholder only on its own, not inside "Candidate Corp".
        assert is_meaningful("Candidate Corp")


class TestNormalizeDate:
    @pytest.mark.parametrize(
        ("raw", "iso"),
        [
            ("15 March 2023", "2023-03-15"),
            ("02/06/2022", "2022-06-02"),
            ("1er janvier 2024", "2024-01-01"),
        ],
    )
    def test_parses_common_formats(self, raw, iso):
        assert normalize_date(raw) == iso

    def test_unparseable_is_returned_unchanged(self):
        assert normalize_date("à la signature") == "à la signature"

    def test_none(self):
        assert normalize_date(None) is None


class TestExtractJsonObject:
    def test_plain_json(self):
        assert extract_json_object('{"a": 1}') == {"a": 1}

    def test_code_fence_and_prose(self):
        text = 'Voici le résultat :\n```json\n{"a": {"b": [1, 2]}}\n```\nMerci.'
        assert extract_json_object(text) == {"a": {"b": [1, 2]}}

    def test_deeply_nested(self):
        text = 'x {"a": {"b": {"c": {"d": 1}}}} y'
        assert extract_json_object(text) == {"a": {"b": {"c": {"d": 1}}}}

    def test_skips_invalid_braces(self):
        assert extract_json_object('{not json} then {"ok": true}') == {"ok": True}

    @pytest.mark.parametrize("text", ["", "no json here", "[1, 2, 3]"])
    def test_none_when_absent(self, text):
        assert extract_json_object(text) is None


class TestContractInfo:
    def test_from_llm_json_cleans_placeholders(self):
        info = ContractInfo.from_llm_json(
            {
                "nom_client": "Non précisé",
                "discloser": "Helios Robotics Inc.",
                "type_contrat": "NDA",
                "date_contrat": "2 June 2022",
                "montant": None,
                "signataires": [
                    {
                        "nom_complet": "Emily Carter",
                        "titre": "General Counsel",
                        "date_signature": "2 June 2022",
                    },
                    {"nom_complet": "name of signatory", "titre": "_____"},
                    "not a dict",
                ],
            }
        )
        assert info.client == "Helios Robotics Inc."
        assert info.contract_type == "NDA"
        assert info.contract_date == "2022-06-02"
        assert info.amount is None
        assert info.signatories == [Signatory("Emily Carter", "General Counsel", "2022-06-02")]

    def test_numeric_amount(self):
        assert ContractInfo.from_llm_json({"montant": 48000}).amount == "48000"

    def test_is_empty(self):
        assert ContractInfo.from_llm_json({"nom_client": "_____", "signataires": []}).is_empty()


class TestFormatAnswer:
    info = ContractInfo(
        client="ACME",
        contract_type="Prestation de services",
        signatories=[Signatory("Jean Dupont", "CEO", "2023-01-05"), Signatory("Anna Roy")],
    )

    def test_single_field(self):
        assert format_answer(self.info, [ContractField.CLIENT]) == "Nom du client : ACME."

    def test_missing_field(self):
        assert "non disponible" in format_answer(self.info, [ContractField.AMOUNT])

    def test_confidential_amount(self):
        info = ContractInfo(amount="Confidentiel")
        assert format_answer(info, [ContractField.AMOUNT]) == "Montant : confidentiel."

    def test_signature_dates_only_lists_dated_signatories(self):
        text = format_answer(self.info, [ContractField.SIGNATURE_DATES])
        assert "Jean Dupont (CEO), signé le 2023-01-05" in text
        assert "Anna Roy" not in text

    def test_empty_field_list_renders_everything(self):
        assert format_answer(self.info, []).count("\n") >= len(ALL_FIELDS) - 1
