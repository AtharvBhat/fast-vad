use ndarray::Array2;
use numpy::{PyArray1, PyArray2, PyReadonlyArray1};
use pyo3::prelude::*;

pub mod vad;

#[pyclass]
struct FeatureExtractor {
    feature_extractor: vad::filterbank::FilterBank,
}

#[pymethods]
impl FeatureExtractor {
    #[new]
    fn new(sample_rate: usize) -> PyResult<Self> {
        if ![8000, 16000].contains(&sample_rate) {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
                "Unsupported sample rate: {} Hz. Only 8000 and 16000 Hz are supported.",
                sample_rate
            )));
        }
        Ok(Self {
            feature_extractor: vad::filterbank::FilterBank::new(sample_rate),
        })
    }

    #[getter]
    fn frame_size(&self) -> usize {
        self.feature_extractor.frame_size()
    }

    #[getter]
    fn hann_window<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f32>> {
        PyArray1::from_slice(py, self.feature_extractor.hann_window())
    }

    fn extract_features<'py>(
        &self,
        py: Python<'py>,
        audio: PyReadonlyArray1<'py, f32>,
    ) -> PyResult<Bound<'py, PyArray2<f32>>> {
        let audio = audio.as_slice()?;
        let features = py.detach(|| self.feature_extractor.compute_filterbank(audio));
        let num_frames = features.len();

        let mut arr = Array2::<f32>::zeros((num_frames, vad::constants::NUM_BANDS));
        for (i, frame) in features.iter().enumerate() {
            arr.row_mut(i).assign(&ndarray::ArrayView1::from(
                &frame.to_array() as &[f32; vad::constants::NUM_BANDS]
            ));
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
