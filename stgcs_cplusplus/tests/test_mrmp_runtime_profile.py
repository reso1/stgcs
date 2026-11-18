import csv
import math
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from experiments.mrmp import MRMPExperiment
from experiments.reports.mrmp_runtime import MRMPWPBSRuntimeBreakdownReport as Report


class RuntimeProfileTests(unittest.TestCase):
    def test_unavailable_native_profile_is_rejected_even_with_all_columns(self):
        for invalid in (math.nan, math.inf, -1):
            with self.subTest(invalid=invalid), TemporaryDirectory() as directory:
                values = dict.fromkeys(MRMPExperiment.RESULT_COLUMNS, 0)
                values.update(name='test', instance_id='test', is_success=True, runtime=10., budget=120.,
                              mp_convex_restriction_runtime=invalid)
                path = Path(directory) / 'results.csv'
                with path.open('w', newline='') as stream:
                    csv.writer(stream).writerow([values[name] for name in MRMPExperiment.RESULT_COLUMNS])
                row = MRMPExperiment.load_result_rows(path)[0]
                self.assertFalse(row['has_disjoint_breakdown_profile'])
                row['has_disjoint_breakdown_profile'] = True
                with self.assertRaisesRegex(ValueError, 'finite, nonnegative'):
                    Report.row_components(row)

    def test_optimization_overlap_is_subtracted_once(self):
        row = dict.fromkeys(MRMPExperiment.RESULT_COLUMNS, 0)
        row.update(runtime=10., pbs_runtime=8., ecd_runtime=1., cc_runtime=.5,
                   mp_gcs_runtime=.5, mp_gub_runtime=2., mp_search_runtime=4.,
                   mp_convex_restriction_runtime=3., mp_dominance_check_runtime=1.,
                   mp_dominance_convex_restriction_runtime=.5,
                   has_disjoint_breakdown_profile=True)
        self.assertEqual(Report.row_components(row),
                         dict(ecd=1., convex_restriction=3., search_procedure=2.5, others=3.5))


if __name__ == '__main__':
    unittest.main()
