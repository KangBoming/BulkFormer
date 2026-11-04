import unittest
import numpy as np
import pandas as pd
from bulkformer.preprocessing import prepare_atlas_input,prepare_latest_input,select_variable_genes


class InputContracts(unittest.TestCase):
    def setUp(self):
        self.norm=pd.DataFrame({'ensembl_gene':['ENSG1','ENSG2','ENSG3'],
                                'gene_length_bp':[1000,2000,1000]})
        self.vocab=pd.DataFrame({'ensembl_gene':['ENSG1','ENSG2'],'matrix_column':[0,1]})

    def test_atlas_denominator_includes_non_input_genes(self):
        counts=pd.DataFrame([[100,100,100],[200,100,100]],columns=self.norm.ensembl_gene)
        expression,_=prepare_atlas_input(counts,self.vocab,self.norm,input_scale='counts')
        np.testing.assert_allclose(expression[0],np.log2(1+np.array([400000.,200000.])),rtol=1e-6)

    def test_missing_and_measured_zero_are_different(self):
        frame=pd.DataFrame([[0,10]],columns=['ENSG1.2','ENSG2'])
        expression,coverage=prepare_latest_input(frame,self.norm,input_scale='tpm')
        self.assertEqual(expression[0,0],0.)
        self.assertEqual(expression[0,2],-10.)
        self.assertEqual(coverage.observed_genes.iloc[0],2)

    def test_log_scales_and_double_normalization(self):
        frame=pd.DataFrame([[0,10,100],[10,100,1]],columns=self.norm.ensembl_gene)
        a,_=prepare_atlas_input(frame,self.vocab,self.norm,input_scale='tpm')
        b,_=prepare_atlas_input(np.log2(1+frame),self.vocab,self.norm,input_scale='log2_tpm')
        np.testing.assert_allclose(a,b,rtol=1e-6)

    def test_duplicate_gene_ids_and_unsupported_arrays_rejected(self):
        frame=pd.DataFrame([[1,2]],columns=['ENSG1.1','ENSG1.2'])
        with self.assertRaises(ValueError):prepare_latest_input(frame,self.norm,input_scale='counts')
        frame.columns=['ENSG1','ENSG2']
        with self.assertRaises(ValueError):prepare_latest_input(frame,self.norm,input_scale='array')

    def test_panel_is_fixed_unique_and_never_filled_with_missing_genes(self):
        expression=np.array([[0,0,3,-10],[2,2,3,-10]],dtype=np.float32)
        np.testing.assert_array_equal(select_variable_genes(expression,2),[0,1])
        np.testing.assert_array_equal(select_variable_genes(expression[::-1],1),[0])
        with self.assertRaises(ValueError):select_variable_genes(expression,3)


if __name__=='__main__':unittest.main()
