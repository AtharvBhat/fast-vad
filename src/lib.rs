use ndarray::Array2;
use numpy::{PyArray2, PyReadonlyArray1};
use pyo3::prelude::*;

pub mod vad;

#[pyclass]
struct FeatureExtractor {
    feature_extractor: vad::filterbank::FilterBank,
}

#[pymethods]
impl FeatureExtractor {
    #[new]
    fn new() -> Self {
        Self {
            feature_extractor: vad::filterbank::FilterBank::new(),
        }
    }

    fn extract_features<'py>(
        &self,
        py: Python<'py>,
        audio: PyReadonlyArray1<'py, f32>,
        sample_rate: u32,
    ) -> PyResult<Bound<'py, PyArray2<f32>>> {
        if sample_rate != 16000 {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
                "Only 16 kHz sample rate is supported, got {}",
                sample_rate
            )));
        }
        let audio = audio.as_slice()?;
        let features = self.feature_extractor.compute_filterbank(audio);
        let num_frames = features.len();

        let mut arr = Array2::<f32>::zeros((num_frames, vad::constants::NUM_BANDS));
        for (i, frame) in features.iter().enumerate() {
            arr.row_mut(i)
                .assign(&ndarray::ArrayView1::from(&frame.to_array()));
        }
        let array = PyArray2::from_owned_array(py, arr);
        Ok(array)
    }
}

#[pymodule]
fn fast_vad(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<FeatureExtractor>()?;
    Ok(())
}
