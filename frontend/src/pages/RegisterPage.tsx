import { useEffect, useState, type FormEvent } from "react";
import { useNavigate } from "react-router-dom";
import { api, ApiError } from "../api/client";
import { useAuth } from "../useAuth";
import type { RegistrationOption, RegistrationOptions } from "../types";
import Button from "../components/ui/Button";

// First-time reviewer registration (DEVIATIONS.md #282). Every field is
// mandatory; the rules below mirror the server's (app.schemas.auth.
// RegistrationIn), which is the one that decides.

type Field =
  | "sex"
  | "cadre"
  | "expertise_level"
  | "years_post_internship"
  | "organisation"
  | "role_title"
  | "safaricom_phone";

const inputClass = "w-full rounded-lg border border-border bg-surface p-2 text-[15px] text-ink";

function RadioField({
  legend,
  name,
  options,
  value,
  onChange,
  error,
}: {
  legend: string;
  name: string;
  options: RegistrationOption[];
  value: string;
  onChange: (v: string) => void;
  error?: string;
}) {
  return (
    <fieldset className="grid gap-1">
      <legend className="mb-1 text-[14px] font-medium text-ink">{legend}</legend>
      {options.map((o) => (
        <label key={o.value} className="flex items-center gap-2 text-[14px] text-ink">
          <input
            type="radio"
            name={name}
            value={o.value}
            checked={value === o.value}
            onChange={() => onChange(o.value)}
            className="h-4 w-4 accent-[var(--color-accent-strong)]"
          />
          {o.label}
        </label>
      ))}
      {error && <p className="text-[12px] text-danger">{error}</p>}
    </fieldset>
  );
}

export default function RegisterPage() {
  const { auth, setRegistrationStatus } = useAuth();
  const resubmitting = auth?.registrationStatus === "rejected";
  const navigate = useNavigate();
  const [options, setOptions] = useState<RegistrationOptions | null>(null);
  const [values, setValues] = useState<Record<Field, string>>({
    sex: "",
    cadre: "",
    expertise_level: "",
    years_post_internship: "",
    organisation: "",
    role_title: "",
    safaricom_phone: "",
  });
  const [errors, setErrors] = useState<Partial<Record<Field, string>>>({});
  const [serverError, setServerError] = useState<string | null>(null);
  const [rejectionReason, setRejectionReason] = useState<string | null>(null);
  const [left, setLeft] = useState<number | null>(null);
  const [busy, setBusy] = useState(false);

  useEffect(() => {
    api.getRegistrationOptions().then(setOptions, (e) =>
      setServerError(e instanceof ApiError ? e.message : String(e)),
    );
  }, []);

  // A rejected reviewer edits their previous answers (DEVIATIONS.md #284).
  useEffect(() => {
    if (!resubmitting) return;
    api.getMe().then(
      (me) => {
        setRejectionReason(me.rejection_reason);
        setLeft(me.resubmissions_left);
      },
      () => undefined,
    );
    api.getOwnRegistration().then(
      (prev) =>
        setValues({
          sex: prev.sex ?? "",
          cadre: prev.cadre ?? "",
          expertise_level: prev.expertise_level ?? "",
          years_post_internship:
            prev.years_post_internship == null ? "" : String(prev.years_post_internship),
          organisation: prev.organisation ?? "",
          role_title: prev.role_title ?? "",
          safaricom_phone: prev.safaricom_phone ?? "",
        }),
      () => undefined, // nothing to prefill: the form simply starts empty
    );
  }, [resubmitting]);

  const set = (field: Field) => (v: string) => setValues((prev) => ({ ...prev, [field]: v }));

  const validate = (o: RegistrationOptions): Partial<Record<Field, string>> => {
    const e: Partial<Record<Field, string>> = {};
    if (!values.sex) e.sex = "Select one.";
    if (!values.cadre) e.cadre = "Select one.";
    if (!values.expertise_level) e.expertise_level = "Select one.";
    const years = values.years_post_internship.trim();
    if (!/^[0-9]+$/.test(years)) e.years_post_internship = "Enter a whole number.";
    else if (Number(years) < o.years_min || Number(years) > o.years_max)
      e.years_post_internship = `Enter a number from ${o.years_min} to ${o.years_max}.`;
    for (const f of ["organisation", "role_title"] as const) {
      const n = values[f].trim().length;
      if (n < o.text_min || n > o.text_max)
        e[f] = `Enter ${o.text_min} to ${o.text_max} characters (now ${n}).`;
    }
    // Optional: only participants in Kenya give one (DEVIATIONS.md #287).
    if (values.safaricom_phone && !/^0[17][0-9]{8}$/.test(values.safaricom_phone))
      e.safaricom_phone = "Enter a 10-digit Kenyan mobile number, e.g. 0712345678, or leave blank.";
    return e;
  };

  const submit = async (ev: FormEvent) => {
    ev.preventDefault();
    if (!options) return;
    const found = validate(options);
    setErrors(found);
    if (Object.keys(found).length) return;
    setBusy(true);
    setServerError(null);
    try {
      const me = await api.submitRegistration({
        sex: values.sex,
        cadre: values.cadre,
        expertise_level: values.expertise_level,
        years_post_internship: Number(values.years_post_internship.trim()),
        organisation: values.organisation.trim(),
        role_title: values.role_title.trim(),
        safaricom_phone: values.safaricom_phone || null,
      });
      setRegistrationStatus(me.registration_status);
      navigate("/pending", { replace: true });
    } catch (e) {
      setServerError(e instanceof ApiError ? e.message : String(e));
    } finally {
      setBusy(false);
    }
  };

  if (resubmitting && left === 0)
    return (
      <section className="mx-auto mt-10 grid max-w-[480px] gap-3">
        <h2 className="text-lg font-semibold text-ink">No resubmissions left</h2>
        <p className="text-[14px] text-ink">
          This registration can't be submitted again. Please contact the study team.
        </p>
        <Button type="button" onClick={() => navigate("/pending", { replace: true })}>
          Back
        </Button>
      </section>
    );

  if (!options)
    return (
      <p className="mt-10 text-[13px] text-ink-muted">{serverError ?? "Loading…"}</p>
    );

  const textField = (field: "organisation" | "role_title", label: string) => (
    <label className="grid gap-1 text-[14px] font-medium text-ink">
      {label}
      <input
        value={values[field]}
        onChange={(e) => set(field)(e.target.value)}
        maxLength={options.text_max}
        className={inputClass}
        required
      />
      <span className="text-[12px] font-normal text-ink-muted">
        {options.text_min} to {options.text_max} characters
      </span>
      {errors[field] && <span className="text-[12px] text-danger">{errors[field]}</span>}
    </label>
  );

  return (
    <section className="mx-auto mt-6 grid max-w-[560px] gap-4">
      <div>
        <h2 className="text-lg font-semibold text-ink">
          {resubmitting ? "Update your registration" : "Reviewer registration"}
        </h2>
        <p className="text-[13px] text-ink-muted">
          {resubmitting
            ? "Your previous answers are filled in. Correct anything that needs it and submit again; an administrator will review it."
            : "All fields are required. An administrator reviews your registration before you can start reviewing."}
        </p>
      </div>
      {rejectionReason && (
        <div className="rounded-lg border border-border bg-surface-alt p-3">
          <p className="text-[12px] font-semibold uppercase tracking-wide text-ink-muted">
            Reason given by the administrator
          </p>
          <p className="mt-1 whitespace-pre-line text-[14px] text-ink">{rejectionReason}</p>
        </div>
      )}
      <form onSubmit={submit} noValidate className="grid gap-5">
        <RadioField
          legend="Sex"
          name="sex"
          options={options.sex}
          value={values.sex}
          onChange={set("sex")}
          error={errors.sex}
        />
        <RadioField
          legend="Clinical cadre"
          name="cadre"
          options={options.cadre}
          value={values.cadre}
          onChange={set("cadre")}
          error={errors.cadre}
        />
        <RadioField
          legend="Expertise level"
          name="expertise_level"
          options={options.expertise_level}
          value={values.expertise_level}
          onChange={set("expertise_level")}
          error={errors.expertise_level}
        />
        <label className="grid gap-1 text-[14px] font-medium text-ink">
          Years of experience after internship
          <input
            inputMode="numeric"
            value={values.years_post_internship}
            onChange={(e) => set("years_post_internship")(e.target.value.replace(/[^0-9]/g, ""))}
            className={`${inputClass} max-w-[120px]`}
            required
          />
          <span className="text-[12px] font-normal text-ink-muted">Whole number</span>
          {errors.years_post_internship && (
            <span className="text-[12px] text-danger">{errors.years_post_internship}</span>
          )}
        </label>
        {textField("organisation", "Current work organisation")}
        {textField("role_title", "Current role / title")}
        <label className="grid gap-1 text-[14px] font-medium text-ink">
          Kenyan Safaricom or Airtel number (optional)
          <input
            inputMode="numeric"
            autoComplete="tel"
            value={values.safaricom_phone}
            maxLength={10}
            onChange={(e) => set("safaricom_phone")(e.target.value.replace(/[^0-9]/g, ""))}
            className={`${inputClass} max-w-[200px]`}
            placeholder="07XXXXXXXX"
          />
          <span className="text-[12px] font-normal text-ink-muted">
            Participants in Kenya only. Used only to send data-bundle reimbursements; leave blank
            if you are outside Kenya.
          </span>
          {errors.safaricom_phone && (
            <span className="text-[12px] text-danger">{errors.safaricom_phone}</span>
          )}
        </label>
        {serverError && (
          <p role="alert" className="text-[13px] text-danger">
            {serverError}
          </p>
        )}
        <Button type="submit" variant="primary" disabled={busy}>
          {busy ? "Submitting…" : resubmitting ? "Resubmit registration" : "Submit registration"}
        </Button>
      </form>
    </section>
  );
}
