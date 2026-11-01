import { fmtProbability } from "@/lib/ui";

/**
 * A single dimension as a labelled bar.
 *
 * The bar is one hue at fixed width with the value always printed beside it:
 * the number is the data, the bar is a fast comparison aid. Nothing is tinted
 * by whether the value is "good".
 */
export function ProbabilityBar({
  label,
  value,
  color,
  width = 96,
}: {
  label: string;
  value: number;
  color: string;
  width?: number;
}) {
  const pct = Math.max(0, Math.min(1, value)) * 100;

  return (
    <div className="flex items-center gap-2.5">
      <span className="w-[86px] shrink-0 truncate text-[11.5px] text-text-secondary">{label}</span>
      <span
        className="relative h-[6px] shrink-0 overflow-hidden rounded-full bg-surface-3"
        style={{ width }}
        role="img"
        aria-label={`${label} ${fmtProbability(value)}`}
      >
        <span
          className="absolute inset-y-0 left-0 rounded-full"
          style={{ width: `${pct}%`, background: color }}
        />
      </span>
      <span className="num w-[34px] shrink-0 text-right text-[11.5px] text-text-primary">
        {fmtProbability(value)}
      </span>
    </div>
  );
}
