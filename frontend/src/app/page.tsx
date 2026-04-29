import ImageUploader from "@/components/ImageUploader";
import ModelInfo from "@/components/ModelInfo";

export default function Home() {
  return (
    <main className="min-h-screen px-4 py-8 text-ink sm:px-6 lg:px-8">
      <div className="mx-auto flex w-full max-w-6xl flex-col gap-8">
        <header className="flex flex-col gap-4 border-b border-slate-200 pb-6">
          <div>
            <p className="mb-2 text-sm font-semibold uppercase tracking-[0.12em] text-clinical">
              APTOS 2019 Blindness Detection
            </p>
            <h1 className="max-w-4xl text-3xl font-bold tracking-normal text-ink sm:text-5xl">
              Diabetic Retinopathy Classifier
            </h1>
          </div>
          <p className="max-w-3xl text-base leading-7 text-slate-600">
            Upload a retinal fundus image and analyze it with the trained ResNet50
            PyTorch model. The system returns the predicted severity class,
            confidence, and class probabilities for all five APTOS labels.
          </p>
        </header>

        <section className="grid gap-6 lg:grid-cols-[minmax(0,1fr)_22rem]">
          <ImageUploader />
          <aside className="flex flex-col gap-4">
            <ModelInfo />
            <div className="rounded-lg border border-amber-200 bg-amber-50 p-4 text-sm leading-6 text-amber-900 shadow-soft">
              This system is a research prototype and does not replace consultation
              with an ophthalmologist.
            </div>
          </aside>
        </section>
      </div>
    </main>
  );
}
