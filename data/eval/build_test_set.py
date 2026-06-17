# -*- coding: utf-8 -*-
"""Değerlendirme test setini üretir (data/eval/test_set.jsonl).

Test seti, klasik ve iyi belgelenmiş ilaç-ilaç / ilaç-hastalık etkileşimlerinden
oluşur. Etiketler (etkileşim var/yok, şiddet, mekanizma) TİTCK Kısa Ürün Bilgisi
(KÜB) belgelerinin "4.5 Diğer tıbbi ürünler ile etkileşimler" bölümünde belgelenen
klasik etkileşimlere dayanır. Her girdi_1 değeri, modelin tanıdığı ilaç/hastalık
listelerinden (src/utils/config.py) seçilmiştir; böylece test seti uçtan uca
çalıştırılabilir.

UYARI: Etiketler eğitim/değerlendirme amaçlıdır, klinik karar için kullanılamaz.

Kullanım:
    python data/eval/build_test_set.py
"""

import json
from pathlib import Path

KAYNAK = "TİTCK KÜB §4.5"

# (girdi_1, girdi_1_tip, girdi_2, etkilesim_var, siddet, mekanizma, ozet)
POZITIF = [
    ("warfarin", "ilac", "aspirin", "major", ["kanama", "antiagregan", "antikoagülan"],
     "Warfarin ile aspirin birlikte kullanımı kanama riskini belirgin şekilde artırır."),
    ("warfarin", "ilac", "ağrı kesici", "major", ["kanama", "NSAİİ", "gastrointestinal"],
     "Warfarin ile NSAİİ türü ağrı kesiciler gastrointestinal kanama riskini artırır."),
    ("warfarin", "ilac", "iltihap kurutucu", "major", ["kanama", "NSAİİ"],
     "Warfarin ile NSAİİ (iltihap kurutucu) birlikte kullanımı kanamaya yol açabilir."),
    ("warfarin", "ilac", "antibiyotik", "orta", ["INR", "kanama"],
     "Bazı antibiyotikler warfarinin INR değerini yükselterek kanama riskini artırır."),
    ("warfarin", "ilac", "parol", "orta", ["INR", "parasetamol"],
     "Yüksek doz parasetamol warfarinin INR değerini yükseltebilir."),
    ("klopidogrel", "ilac", "aspirin", "major", ["kanama", "antiagregan"],
     "Klopidogrel ve aspirin (ikili antiagregan) kanama riskini artırır."),
    ("klopidogrel", "ilac", "mide koruyucu", "orta", ["omeprazol", "etkinlik", "CYP2C19"],
     "Omeprazol gibi mide koruyucular klopidogrelin antiagregan etkinliğini azaltabilir."),
    ("aspirin", "ilac", "ağrı kesici", "orta", ["gastrointestinal", "ülser", "kanama"],
     "Aspirin ile diğer NSAİİ ağrı kesiciler ülser ve GİS kanama riskini artırır."),
    ("aspirin", "ilac", "iltihap kurutucu", "orta", ["gastrointestinal", "kanama"],
     "Aspirin ile NSAİİ birlikte kullanımı gastrointestinal kanamayı artırır."),
    ("aspirin", "ilac", "kan sulandırıcı", "major", ["kanama"],
     "Aspirin ile kan sulandırıcı birlikte kullanımı kanama riskini artırır."),
    ("kan sulandırıcı", "ilac", "ağrı kesici", "major", ["kanama", "NSAİİ"],
     "Kan sulandırıcı ile NSAİİ ağrı kesici kanama riskini ciddi şekilde artırır."),
    ("kan sulandırıcı", "ilac", "aspirin", "major", ["kanama"],
     "Kan sulandırıcı ile aspirin birlikte kullanımı kanamaya yol açabilir."),
    ("miğren", "ilac", "antidepresan", "major", ["serotonin sendromu", "triptan", "SSRI"],
     "Triptan (miğren ilacı) ile SSRI antidepresan serotonin sendromuna yol açabilir."),
    ("antidepresan", "ilac", "sakinleştirici", "orta", ["sedasyon", "MSS"],
     "Antidepresan ile sakinleştirici birlikte kullanımı merkezi sinir sistemi depresyonunu artırır."),
    ("antidepresan", "ilac", "sinir ilacı", "orta", ["serotonin", "MSS"],
     "Antidepresan ile sinir ilacı serotonerjik ve MSS yan etkilerini artırabilir."),
    ("potasyum", "ilac", "idrar söktürücü", "major", ["hiperkalemi", "potasyum"],
     "Potasyum tutucu idrar söktürücü ile potasyum hiperkalemiye yol açabilir."),
    ("potasyum takviyesi", "ilac", "idrar söktürücü", "major", ["hiperkalemi", "potasyum"],
     "Potasyum takviyesi ile potasyum tutucu diüretik hiperkalemi riskini artırır."),
    ("barbitürat", "ilac", "sakinleştirici", "major", ["solunum depresyonu", "MSS"],
     "Barbitürat ile sakinleştirici birlikte solunum depresyonuna yol açabilir."),
    ("barbitürat", "ilac", "antidepresan", "orta", ["MSS", "enzim"],
     "Barbitürat ile antidepresan MSS depresyonu ve enzim indüksiyonuna neden olabilir."),
    ("anestezi", "ilac", "sakinleştirici", "major", ["solunum depresyonu", "MSS"],
     "Anestezik ile sakinleştirici additif MSS ve solunum depresyonu yapar."),
    ("kas gevşetici", "ilac", "sakinleştirici", "orta", ["sedasyon", "solunum"],
     "Kas gevşetici ile sakinleştirici additif sedasyon ve solunum baskılanması yapar."),
    ("kas gevşetici", "ilac", "barbitürat", "orta", ["MSS"],
     "Kas gevşetici ile barbitürat merkezi sinir sistemi depresyonunu artırır."),
    ("adrenalin", "ilac", "antidepresan", "orta", ["hipertansif", "kan basıncı"],
     "Adrenalin ile bazı antidepresanlar (MAOI/TCA) hipertansif yanıta yol açabilir."),
    ("parol", "ilac", "ateş düşürücü", "orta", ["parasetamol", "doz", "karaciğer"],
     "Parol ile ateş düşürücü çoğunlukla parasetamol içerir; doz aşımı karaciğer hasarı yapabilir."),
    ("metformin", "ilac", "idrar söktürücü", "orta", ["kan şekeri", "glisemik"],
     "İdrar söktürücüler glisemik kontrolü bozarak metformin etkisini etkileyebilir."),
    ("antibiyotik", "ilac", "hormon", "orta", ["kontraseptif", "etkinlik"],
     "Bazı antibiyotikler oral kontraseptif (hormon) etkinliğini azaltabilir."),
    ("hipertansiyon", "hastalik", "grip ilacı", "major", ["kan basıncı", "psödoefedrin", "dekonjestan"],
     "Hipertansiyon hastalarında psödoefedrin içeren grip ilaçları kan basıncını yükseltir."),
    ("tansiyon", "hastalik", "grip ilacı", "major", ["kan basıncı", "psödoefedrin"],
     "Tansiyon hastalarında dekonjestan içeren grip ilaçları kan basıncını yükseltir."),
    ("hipertansiyon", "hastalik", "ağrı kesici", "orta", ["kan basıncı", "NSAİİ", "sıvı"],
     "NSAİİ ağrı kesiciler sıvı tutulumu yaparak kan basıncını yükseltir."),
    ("tansiyon", "hastalik", "ağrı kesici", "orta", ["kan basıncı", "NSAİİ"],
     "NSAİİ ağrı kesiciler tansiyon kontrolünü zorlaştırabilir."),
    ("hipertansiyon", "hastalik", "iltihap kurutucu", "orta", ["kan basıncı", "NSAİİ"],
     "NSAİİ (iltihap kurutucu) hipertansiyonda kan basıncını yükseltebilir."),
    ("tansiyon", "hastalik", "potasyum", "major", ["hiperkalemi", "potasyum"],
     "ACE inhibitörü kullanan tansiyon hastalarında potasyum hiperkalemiye yol açar."),
    ("tansiyon", "hastalik", "potasyum takviyesi", "major", ["hiperkalemi", "potasyum"],
     "Tansiyon ilaçları ile potasyum takviyesi hiperkalemi riskini artırır."),
    ("kalp", "hastalik", "grip ilacı", "orta", ["aritmi", "kan basıncı"],
     "Kalp hastalarında dekonjestan içeren grip ilaçları aritmi ve kan basıncı artışı yapabilir."),
    ("böbrek", "hastalik", "ağrı kesici", "major", ["böbrek", "nefrotoksik", "NSAİİ"],
     "NSAİİ ağrı kesiciler böbrek hastalarında nefrotoksiktir."),
    ("böbrek", "hastalik", "iltihap kurutucu", "major", ["böbrek", "NSAİİ"],
     "NSAİİ (iltihap kurutucu) böbrek fonksiyonlarını kötüleştirebilir."),
    ("astım", "hastalik", "aspirin", "major", ["bronkospazm", "astım"],
     "Aspirin duyarlı astımlılarda aspirin bronkospazma yol açabilir."),
    ("astım", "hastalik", "ağrı kesici", "orta", ["bronkospazm", "NSAİİ"],
     "NSAİİ ağrı kesiciler duyarlı astımlılarda bronkospazm yapabilir."),
    ("karaciğer", "hastalik", "parol", "major", ["karaciğer", "parasetamol", "hepatotoksik"],
     "Karaciğer hastalarında parasetamol (parol) hepatotoksisite riskini artırır."),
    ("gut", "hastalik", "idrar söktürücü", "orta", ["ürik asit", "gut"],
     "İdrar söktürücüler ürik asidi yükselterek gut ataklarını tetikleyebilir."),
]

NEGATIF = [
    ("aspirin", "ilac", "parol", ["parasetamol"],
     "Aspirin ve parasetamol (parol) birlikte yaygın kullanılır; anlamlı etkileşim beklenmez."),
    ("parol", "ilac", "antibiyotik",  [],
     "Parasetamol ile çoğu antibiyotik arasında klinik olarak anlamlı etkileşim beklenmez."),
    ("metformin", "ilac", "parol", [],
     "Metformin ile parasetamol arasında anlamlı bir ilaç-ilaç etkileşimi beklenmez."),
    ("parol", "ilac", "mide koruyucu", [],
     "Parasetamol ile mide koruyucular arasında anlamlı etkileşim beklenmez."),
    ("ateş düşürücü", "ilac", "mide koruyucu", [],
     "Parasetamol türü ateş düşürücü ile mide koruyucu arasında anlamlı etkileşim beklenmez."),
    ("alerji", "ilac", "parol", [],
     "Alerji ilacı (antihistaminik) ile parasetamol arasında anlamlı etkileşim beklenmez."),
    ("nezle", "hastalik", "parol", [],
     "Nezle durumunda parasetamol güvenle kullanılır; anlamlı etkileşim beklenmez."),
    ("grip", "hastalik", "parol", [],
     "Grip durumunda parasetamol güvenle kullanılır; anlamlı etkileşim beklenmez."),
    ("diyabet", "hastalik", "parol", [],
     "Diyabet hastalarında parasetamol ile anlamlı bir etkileşim beklenmez."),
    ("hipertansiyon", "hastalik", "parol", [],
     "Hipertansiyonda parasetamol tercih edilen analjeziktir; anlamlı etkileşim beklenmez."),
    ("tansiyon", "hastalik", "parol", [],
     "Tansiyon hastalarında parasetamol genellikle güvenlidir; anlamlı etkileşim beklenmez."),
    ("kansızlık", "hastalik", "parol", [],
     "Kansızlık (anemi) durumunda parasetamol ile anlamlı etkileşim beklenmez."),
    ("obezite", "hastalik", "parol", [],
     "Obezitede parasetamol ile anlamlı bir ilaç etkileşimi beklenmez."),
    ("astım", "hastalik", "parol", [],
     "Astımda parasetamol NSAİİ'ye tercih edilir; anlamlı etkileşim beklenmez."),
    ("zatürre", "hastalik", "parol", [],
     "Zatürre durumunda parasetamol ile anlamlı bir etkileşim beklenmez."),
    ("romatizma", "hastalik", "mide koruyucu", [],
     "Romatizma hastalarında mide koruyucu NSAİİ ile birlikte güvenle kullanılır."),
    ("nezle", "hastalik", "antibiyotik", [],
     "Nezlede antibiyotik ile parasetamol/destek tedavisi arasında anlamlı etkileşim beklenmez."),
    ("grip", "hastalik", "mide koruyucu", [],
     "Grip durumunda mide koruyucu ile anlamlı bir etkileşim beklenmez."),
    ("zatürre", "hastalik", "mide koruyucu", [],
     "Zatürre durumunda mide koruyucu ile anlamlı bir etkileşim beklenmez."),
    ("kalp", "hastalik", "parol", [],
     "Kalp hastalarında parasetamol genellikle güvenli analjeziktir; anlamlı etkileşim beklenmez."),
]


def build():
    rows = []
    idx = 1
    for girdi_1, tip, girdi_2, siddet, mek, ozet in POZITIF:
        rows.append({
            "id": f"pos-{idx:03d}",
            "girdi_1": girdi_1,
            "girdi_1_tip": tip,
            "girdi_2": girdi_2,
            "etkilesim_var": True,
            "siddet": siddet,
            "mekanizma": mek,
            "referans_ozet": ozet,
            "kaynak": KAYNAK,
        })
        idx += 1
    idx = 1
    for girdi_1, tip, girdi_2, mek, ozet in NEGATIF:
        rows.append({
            "id": f"neg-{idx:03d}",
            "girdi_1": girdi_1,
            "girdi_1_tip": tip,
            "girdi_2": girdi_2,
            "etkilesim_var": False,
            "siddet": "yok",
            "mekanizma": mek,
            "referans_ozet": ozet,
            "kaynak": KAYNAK,
        })
        idx += 1
    return rows


def main():
    rows = build()
    out_path = Path(__file__).resolve().parent / "test_set.jsonl"
    with open(out_path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    pos = sum(1 for r in rows if r["etkilesim_var"])
    neg = len(rows) - pos
    print(f"Yazıldı: {out_path}")
    print(f"Toplam {len(rows)} örnek ({pos} etkileşim var, {neg} etkileşim yok)")


if __name__ == "__main__":
    main()
