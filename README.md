# Προσομοίωση Αξιοπιστίας Σύνθετου Συστήματος

Υπολογιστική Εργασία - Αξιοπιστία Συστημάτων

## Description

Προσομοίωση λειτουργίας συστήματος με 7 εξαρτήματα (C1-C7) και εξαγωγή συμπερασμάτων για την αξιοπιστία του συστήματος με βάση μεθόδους Monte Carlo.

### Section 1: System without Repair
1. **Part 1**: Προσομοίωση βλαβών κάθε εξαρτήματος (χωρίς MTTR)
2. **Part 2**: Υπολογισμός πειραματικών τιμών λ, R, MTTF για το σύστημα

### Section 2: System with Repair
3. **Part 3**: Προσομοίωση βλαβών και επιδιορθώσεων κάθε εξαρτήματος
4. **Part 4**: Υπολογισμός MTBF, MUT, MTTR, Availability για το σύστημα

## Execution Steps

```bash
pip install -r requirements.txt

python main.py

# Execute specific part
python main.py --part 1 2
```

## Results

Γραφήματα και στατιστικά αποθηκεύονται στον φάκελο `results/` για κάθε μέρος της εργασίας.
