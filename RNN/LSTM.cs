using System;

namespace DeepLearning
{
    /// <summary>
    /// پیاده‌سازی ساده LSTM برای یادگیری دنباله‌های زمانی
    /// </summary>
    public class SimpleLSTM
    {
        private int inputSize;
        private int hiddenSize;
        
        // وزن‌های Forget Gate
        private double[,] Wf;
        private double[] bf;
        
        // وزن‌های Input Gate
        private double[,] Wi;
        private double[] bi;
        
        // وزن‌های Cell State Candidate
        private double[,] Wc;
        private double[] bc;
        
        // وزن‌های Output Gate
        private double[,] Wo;
        private double[] bo;
        
        // وزن‌های خروجی نهایی
        private double[,] Wy;
        private double[] by;
        
        private Random random;

        /// <summary>
        /// سازنده کلاس LSTM
        /// </summary>
        /// <param name="inputSize">اندازه ورودی</param>
        /// <param name="hiddenSize">اندازه لایه پنهان</param>
        public SimpleLSTM(int inputSize = 1, int hiddenSize = 4)
        {
            this.inputSize = inputSize;
            this.hiddenSize = hiddenSize;
            this.random = new Random(42); // برای تکرارپذیری نتایج
            
            InitializeWeights();
            
            Console.WriteLine("🧠 LSTM آماده شد!");
            Console.WriteLine($"📊 اندازه ورودی: {inputSize}, اندازه حافظه مخفی: {hiddenSize}");
        }

        /// <summary>
        /// مقداردهی اولیه وزن‌ها
        /// </summary>
        private void InitializeWeights()
        {
            int combinedSize = inputSize + hiddenSize;
            
            // وزن‌های Forget Gate
            Wf = RandomMatrix(hiddenSize, combinedSize, 0.5);
            bf = FillArray(hiddenSize, 0.1);
            
            // وزن‌های Input Gate
            Wi = RandomMatrix(hiddenSize, combinedSize, 0.5);
            bi = FillArray(hiddenSize, 0.1);
            
            // وزن‌های Cell State Candidate
            Wc = RandomMatrix(hiddenSize, combinedSize, 0.5);
            bc = new double[hiddenSize];
            
            // وزن‌های Output Gate
            Wo = RandomMatrix(hiddenSize, combinedSize, 0.5);
            bo = FillArray(hiddenSize, 0.1);
            
            // وزن‌های خروجی نهایی
            Wy = RandomMatrix(1, hiddenSize, 0.5);
            by = new double[1];
        }

        /// <summary>
        /// تابع فعال‌سازی sigmoid
        /// </summary>
        private double Sigmoid(double x)
        {
            x = Math.Max(-500, Math.Min(500, x)); // محدود کردن برای جلوگیری از overflow
            return 1.0 / (1.0 + Math.Exp(-x));
        }

        /// <summary>
        /// تابع فعال‌سازی tanh
        /// </summary>
        private double Tanh(double x)
        {
            return Math.Tanh(x);
        }

        /// <summary>
        /// ایجاد ماتریس تصادفی
        /// </summary>
        private double[,] RandomMatrix(int rows, int cols, double scale)
        {
            double[,] matrix = new double[rows, cols];
            for (int i = 0; i < rows; i++)
            {
                for (int j = 0; j < cols; j++)
                {
                    // تولید عدد تصادفی با توزیع نرمال (تقریبی)
                    double u1 = random.NextDouble();
                    double u2 = random.NextDouble();
                    double randNormal = Math.Sqrt(-2.0 * Math.Log(u1)) * Math.Sin(2.0 * Math.PI * u2);
                    matrix[i, j] = randNormal * scale;
                }
            }
            return matrix;
        }

        /// <summary>
        /// پر کردن آرایه با مقدار ثابت
        /// </summary>
        private double[] FillArray(int size, double value)
        {
            double[] array = new double[size];
            Array.Fill(array, value);
            return array;
        }

        /// <summary>
        /// ترکیب دو آرایه
        /// </summary>
        private double[] ConcatenateArrays(double[] arr1, double[] arr2)
        {
            double[] result = new double[arr1.Length + arr2.Length];
            Array.Copy(arr1, 0, result, 0, arr1.Length);
            Array.Copy(arr2, 0, result, arr1.Length, arr2.Length);
            return result;
        }

        /// <summary>
        /// ضرب ماتریس در بردار
        /// </summary>
        private double[] MatrixVectorMultiply(double[,] matrix, double[] vector)
        {
            int rows = matrix.GetLength(0);
            int cols = matrix.GetLength(1);
            double[] result = new double[rows];
            
            for (int i = 0; i < rows; i++)
            {
                double sum = 0;
                for (int j = 0; j < cols; j++)
                {
                    sum += matrix[i, j] * vector[j];
                }
                result[i] = sum;
            }
            return result;
        }

        /// <summary>
        /// جمع دو بردار
        /// </summary>
        private double[] AddVectors(double[] v1, double[] v2)
        {
            double[] result = new double[v1.Length];
            for (int i = 0; i < v1.Length; i++)
            {
                result[i] = v1[i] + v2[i];
            }
            return result;
        }

        /// <summary>
        /// ضرب عنصری دو بردار
        /// </summary>
        private double[] ElementwiseMultiply(double[] v1, double[] v2)
        {
            double[] result = new double[v1.Length];
            for (int i = 0; i < v1.Length; i++)
            {
                result[i] = v1[i] * v2[i];
            }
            return result;
        }

        /// <summary>
        /// اعمال تابع sigmoid روی بردار
        /// </summary>
        private double[] ApplySigmoid(double[] vector)
        {
            double[] result = new double[vector.Length];
            for (int i = 0; i < vector.Length; i++)
            {
                result[i] = Sigmoid(vector[i]);
            }
            return result;
        }

        /// <summary>
        /// اعمال تابع tanh روی بردار
        /// </summary>
        private double[] ApplyTanh(double[] vector)
        {
            double[] result = new double[vector.Length];
            for (int i = 0; i < vector.Length; i++)
            {
                result[i] = Tanh(vector[i]);
            }
            return result;
        }

        /// <summary>
        /// یک گام پیش‌رو در LSTM
        /// </summary>
        public (double output, double[] newHidden, double[] newCell, GateInfo gates) ForwardStep(
            double[] input, double[] hPrev, double[] cPrev, bool verbose = false)
        {
            // ترکیب ورودی فعلی با حافظه قبلی
            double[] combined = ConcatenateArrays(input, hPrev);
            
            // 🚪 Forget Gate - تصمیم گیری برای فراموشی
            double[] fGate = ApplySigmoid(AddVectors(MatrixVectorMultiply(Wf, combined), bf));
            
            // 🔄 Input Gate - تصمیم گیری برای ورودی جدید
            double[] iGate = ApplySigmoid(AddVectors(MatrixVectorMultiply(Wi, combined), bi));
            
            // 📝 Cell State Candidate - مقادیر کاندید جدید
            double[] cTilde = ApplyTanh(AddVectors(MatrixVectorMultiply(Wc, combined), bc));
            
            // 🧠 بروزرسانی Cell State
            double[] cNew = AddVectors(ElementwiseMultiply(fGate, cPrev), ElementwiseMultiply(iGate, cTilde));
            
            // 🚪 Output Gate - تصمیم گیری برای خروجی
            double[] oGate = ApplySigmoid(AddVectors(MatrixVectorMultiply(Wo, combined), bo));
            
            // 📤 Hidden State جدید
            double[] hNew = ElementwiseMultiply(oGate, ApplyTanh(cNew));
            
            // 📈 خروجی نهایی
            double output = MatrixVectorMultiply(Wy, hNew)[0] + by[0];
            
            if (verbose)
            {
                Console.WriteLine($"  🔹 Forget Gate: [{string.Join(", ", Array.ConvertAll(fGate, x => x.ToString("F3")))}]");
                Console.WriteLine($"  🔹 Input Gate: [{string.Join(", ", Array.ConvertAll(iGate, x => x.ToString("F3")))}]");
                Console.WriteLine($"  🔹 Output Gate: [{string.Join(", ", Array.ConvertAll(oGate, x => x.ToString("F3")))}]");
                Console.WriteLine($"  🔹 Cell State: [{string.Join(", ", Array.ConvertAll(cNew, x => x.ToString("F3")))}]");
                Console.WriteLine($"  🔹 Hidden State: [{string.Join(", ", Array.ConvertAll(hNew, x => x.ToString("F3")))}]");
                Console.WriteLine($"  🔹 پیش‌بینی: {output:F3}");
            }
            
            var gates = new GateInfo(fGate, iGate, oGate, cTilde);
            return (output, hNew, cNew, gates);
        }

        /// <summary>
        /// پیش‌بینی یک دنباله کامل
        /// </summary>
        public (double[] predictions, GateInfo[] gatesInfo) PredictSequence(double[] inputSequence, bool verbose = false)
        {
            // اولیه‌سازی حافظه
            double[] hCurrent = new double[hiddenSize];
            double[] cCurrent = new double[hiddenSize];
            
            double[] predictions = new double[inputSequence.Length];
            GateInfo[] gatesInfo = new GateInfo[inputSequence.Length];
            
            Console.WriteLine($"🚀 شروع پیش‌بینی دنباله با {inputSequence.Length} عنصر");
            
            for (int i = 0; i < inputSequence.Length; i++)
            {
                if (verbose)
                {
                    Console.WriteLine($"\n📍 گام {i + 1}: ورودی = {inputSequence[i]:F3}");
                }
                
                double[] input = { inputSequence[i] };
                var result = ForwardStep(input, hCurrent, cCurrent, verbose);
                
                predictions[i] = result.output;
                gatesInfo[i] = result.gates;
                hCurrent = result.newHidden;
                cCurrent = result.newCell;
            }
            
            return (predictions, gatesInfo);
        }

        /// <summary>
        /// محاسبه میانگین یک آرایه
        /// </summary>
        private double Average(double[] array)
        {
            double sum = 0;
            foreach (double value in array)
            {
                sum += value;
            }
            return sum / array.Length;
        }

        /// <summary>
        /// اجرای مثال کامل
        /// </summary>
        public static void RunExample()
        {
            Console.WriteLine("=" + new string('=', 49));
            Console.WriteLine("🔢 مثال LSTM برای پیش‌بینی دنباله حسابی");
            Console.WriteLine("=" + new string('=', 49));
            
            // 1️⃣ تولید دیتا
            Console.WriteLine("\n1️⃣ تولید دیتای آموزشی:");
            double[] sequence = GenerateArithmeticSequence(start: 1, step: 3, length: 8);
            Console.WriteLine($"دنباله اصلی: [{string.Join(", ", sequence)}]");
            
            // 2️⃣ ایجاد مدل
            Console.WriteLine("\n2️⃣ ایجاد مدل LSTM:");
            var lstm = new SimpleLSTM(inputSize: 1, hiddenSize: 4);
            
            // 3️⃣ تست پیش‌بینی
            Console.WriteLine("\n3️⃣ تست پیش‌بینی با جزئیات:");
            double[] testSequence = { 1, 4, 7 }; // بخشی از دنباله برای تست
            Console.WriteLine($"دنباله تست: [{string.Join(", ", testSequence)}]");
            
            var (predictions, gatesInfo) = lstm.PredictSequence(testSequence, verbose: true);
            
            // 4️⃣ نمایش نتایج
            Console.WriteLine("\n4️⃣ خلاصه نتایج:");
            Console.WriteLine($"ورودی‌ها: [{string.Join(", ", testSequence)}]");
            Console.WriteLine($"پیش‌بینی‌ها: [{string.Join(", ", Array.ConvertAll(predictions, x => x.ToString("F3")))}]");
            
            // پیش‌بینی عنصر بعدی
            double nextPrediction = predictions[predictions.Length - 1];
            double expectedNext = 10; // عنصر بعدی واقعی در دنباله
            Console.WriteLine($"\n🎯 پیش‌بینی عنصر بعدی: {nextPrediction:F3}");
            Console.WriteLine($"🎯 مقدار واقعی: {expectedNext}");
            Console.WriteLine($"🎯 خطا: {Math.Abs(nextPrediction - expectedNext):F3}");
            
            // 5️⃣ تحلیل گیت‌ها
            Console.WriteLine("\n5️⃣ تحلیل عملکرد گیت‌ها:");
            var lstm_instance = new SimpleLSTM();
            for (int i = 0; i < gatesInfo.Length; i++)
            {
                var gates = gatesInfo[i];
                Console.WriteLine($"گام {i + 1}:");
                Console.WriteLine($"  🔴 میانگین Forget Gate: {lstm_instance.Average(gates.ForgetGate):F3} (بالا = حفظ حافظه)");
                Console.WriteLine($"  🟢 میانگین Input Gate: {lstm_instance.Average(gates.InputGate):F3} (بالا = پذیرش ورودی)");
                Console.WriteLine($"  🔵 میانگین Output Gate: {lstm_instance.Average(gates.OutputGate):F3} (بالا = خروجی فعال)");
            }
            
            Console.WriteLine("\n" + new string('=', 50));
            Console.WriteLine("✅ تمام! حالا می‌تونی ببینی LSTM چطوری کار می‌کنه");
            Console.WriteLine("💡 برای یادگیری واقعی، باید وزن‌ها با backpropagation بروزرسانی بشن");
            Console.WriteLine(new string('=', 50));
        }

        /// <summary>
        /// تولید دنباله حسابی
        /// </summary>
        public static double[] GenerateArithmeticSequence(double start = 1, double step = 2, int length = 10)
        {
            double[] sequence = new double[length];
            for (int i = 0; i < length; i++)
            {
                sequence[i] = start + i * step;
            }
            return sequence;
        }
    }

    /// <summary>
    /// کلاس برای نگهداری اطلاعات گیت‌ها
    /// </summary>
    public class GateInfo
    {
        public double[] ForgetGate { get; }
        public double[] InputGate { get; }
        public double[] OutputGate { get; }
        public double[] CellCandidate { get; }

        public GateInfo(double[] forgetGate, double[] inputGate, double[] outputGate, double[] cellCandidate)
        {
            ForgetGate = forgetGate;
            InputGate = inputGate;
            OutputGate = outputGate;
            CellCandidate = cellCandidate;
        }
    }

    /// <summary>
    /// کلاس اصلی برای اجرای برنامه
    /// </summary>
    public class Program
    {
        public static void Main(string[] args)
        {
            Console.OutputEncoding = System.Text.Encoding.UTF8;
            SimpleLSTM.RunExample();
            
            Console.WriteLine("\nبرای خروج کلیدی بزنید...");
            Console.ReadKey();
        }
    }
}