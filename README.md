Layer Sequence Extraction from Optimized DNNs using Side-Channel Traces
⚙️ A two-phase deep learning pipeline to reconstruct internal DNN layer sequences (CNN/RNN) from GPU kernel traces — with integration of TVM-based optimization and attention visualization.

🚀 Features
✅ Generates 8000 CNN & RNN models with randomized architectures

✅ Simulates kernel traces via Apache TVM

✅ Optimizes CNNs using TVM (Relay / AutoTVM)

✅ Phase 1: Predict operation types (OPi) from traces

✅ Phase 2: Predict full DNN layer sequence from OPi

✅ Evaluation: LER, MAE, F1 metrics

✅ CLI + Full Pipeline runner

✅ Attention map visualization (Multi-Head Attention)

✅ Modular & extensible codebase

🧠 Architecture
1️⃣ Dataset Generation
2️⃣ Trace Simulation (TVM)
3️⃣ Phase 1: Trace → OPi
4️⃣ Phase 2: OPi → Layer Sequence
5️⃣ Evaluation & Visualization

📁 Folder Structure
kotlin
Copy
Edit
layer-sequence-extraction/
├── generator/
│   └── generate_dnn_dataset.py
├── trace_simulator/
│   └── simulate_kernel_trace.py
├── optimization/
│   └── optimize_with_tvm.py
├── phase1/
│   ├── model.py
│   ├── utils.py
│   └── train.py
├── phase2/
│   ├── model.py
│   ├── utils.py
│   ├── train.py
│   └── predict.py
├── utils/
│   ├── evaluation.py
│   ├── constants.py
│   └── visualize_attention.py
├── cli.py
├── run_pipeline.py
└── README.md ← this file
🧪 How to Run
💡 Activate virtual environment first:

bash
Copy
Edit
source tvm-env/bin/activate
export PYTHONPATH=$PYTHONPATH:/path/to/tvm/python
🔧 Step-by-step:

Generate models & traces:

bash
Copy
Edit
python3 generator/generate_dnn_dataset.py
python3 trace_simulator/simulate_kernel_trace.py
Optimize CNNs with TVM:

bash
Copy
Edit
python3 optimization/optimize_with_tvm.py
Phase 1 training:

bash
Copy
Edit
python3 phase1/train.py
Phase 2 training:

bash
Copy
Edit
python3 phase2/train.py
Prediction:

bash
Copy
Edit
python3 phase2/predict.py
Evaluate:

bash
Copy
Edit
python3 cli.py --phase 2 --task evaluate --weights phase2_trained.pth
Full pipeline:

bash
Copy
Edit
python3 run_pipeline.py
📊 Metrics
Label Error Rate (LER)

Mean Absolute Error (MAE)

F1 Score

🔍 Visualization
Use:

bash
Copy
Edit
python3 utils/visualize_attention.py
To visualize attention maps from Phase 2 decoder.

📚 Dependencies
Python 3.10+

PyTorch

TVM (built from source)

scikit-learn

matplotlib