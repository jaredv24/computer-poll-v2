// Called by Vercel Cron (see vercel.json). Triggers a fresh deploy through a Deploy Hook,
// and each deploy rebuilds the poll data during its build step.
module.exports = async (req, res) => {
  const secret = process.env.CRON_SECRET;
  if (secret && req.headers.authorization !== `Bearer ${secret}`) {
    return res.status(401).json({ ok: false, error: "Unauthorized" });
  }
  const hook = process.env.DEPLOY_HOOK_URL;
  if (!hook) {
    return res.status(500).json({ ok: false, error: "DEPLOY_HOOK_URL is not set" });
  }
  const r = await fetch(hook, { method: "POST" });
  const body = await r.text();
  return res.status(r.ok ? 200 : 502).json({ ok: r.ok, status: r.status, body: body.slice(0, 500) });
};
