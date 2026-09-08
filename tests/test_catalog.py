from pathlib import Path

from posefit.catalog import design_id, image_sort_key, split_variant


def test_variant_suffix_stripped():
    assert split_variant("Куртка стеганая (3)") == ("Куртка стеганая", 3)
    assert split_variant("Куртка стеганая") == ("Куртка стеганая", 1)


def test_design_id_merges_colourways():
    assert design_id("Джинсы широкие (19)") == design_id("Джинсы широкие")


def test_design_id_is_case_and_yo_insensitive():
    assert design_id("Платье-СВИТЕР Ёлочка") == design_id("платье-свитер елочка")


def test_plus_size_stays_separate():
    # Разные линейки: сливать их вслепую нельзя, связность ловится по phash.
    assert design_id("PLUS SIZE Куртка-парка") != design_id("Куртка-парка")


def test_image_order_handles_both_naming_schemes():
    names = ["10_fs.webp", "fs.webp", "2_fs.webp"]
    ordered = sorted((Path(n) for n in names), key=image_sort_key)
    assert [p.name for p in ordered] == ["fs.webp", "2_fs.webp", "10_fs.webp"]


def test_rel_posix_normalises_to_nfc():
    import unicodedata

    from posefit.catalog import rel_posix

    # macOS отдаёт 'й' разложенным; хэш от такого пути отличался бы от
    # линуксового, и препроцессинг с сервера не нашёлся бы локально.
    decomposed = unicodedata.normalize("NFD", "Джинсы с высокой посадкой/1.webp")
    got = rel_posix(Path("/data") / decomposed, Path("/data"))
    assert got == unicodedata.normalize("NFC", "Джинсы с высокой посадкой/1.webp")
    assert unicodedata.normalize("NFC", got) == got
