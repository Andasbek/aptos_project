import ImageUploader from "@/components/ImageUploader";
import LanguageSelector from "@/components/LanguageSelector";
import LocalizedHeader from "@/components/LocalizedHeader";
import LocalizedDisclaimer from "@/components/LocalizedDisclaimer";
import ModelInfo from "@/components/ModelInfo";

export default function Home() {
  return (
    <main className="min-h-screen px-4 py-8 text-ink sm:px-6 lg:px-8">
      <div className="mx-auto flex w-full max-w-6xl flex-col gap-8">
        <header className="flex flex-col gap-4 border-b border-slate-200 pb-6">
          <div className="flex flex-col gap-4 sm:flex-row sm:items-start sm:justify-between">
            <LocalizedHeader />
            <LanguageSelector />
          </div>
        </header>

        <section className="grid gap-6 lg:grid-cols-[minmax(0,1fr)_22rem]">
          <ImageUploader />
          <aside className="flex flex-col gap-4">
            <ModelInfo />
            <LocalizedDisclaimer />
          </aside>
        </section>
      </div>
    </main>
  );
}
