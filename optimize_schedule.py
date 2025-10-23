import numpy as np
from sim_ann import SAConfig, SimulatedAnnealer
from astropy.time import Time
from astropy.coordinates import SkyCoord, EarthLocation, AltAz, get_sun
import astropy.units as u
import argparse
import pandas as pd

def get_intercept_unc_squared(x):
    # assume x, y are 1D arrays (possibly containing np.nan)
    mask = np.isfinite(x)
    xv = x[mask]

    n = len(xv)
    sum_xx = sum(xv**2.)
    sum_x = sum(xv)

    return sum_xx/(n*sum_xx - sum_x**2.)


def get_airmass(target, t_mid):
    altaz_frame = AltAz(location=maunakea, obstime=t_mid,
                    pressure=615*u.hPa,  # ~ 0.61 atm at 4.2 km
                    temperature=0*u.deg_C, relative_humidity=0.2,
                    obswl=0.6*u.micron)  # representative wavelength
    
    altaz = target.transform_to(altaz_frame)
    airmass_secz = altaz.secz.value
    airmass = np.where(altaz.alt < 25*u.deg, np.inf, airmass_secz)
    #airmass = np.where(airmass < args.airmass_max, np.inf, airmass)
    return float(airmass)


def fitness(sequence, verbose = False):
    if verbose:
        print(sequence)

    t_boundaries = [Time(args.start, scale="utc")]
    
    for star_id in sequence:
        t_boundaries.append(t_boundaries[-1] + all_targets["service_s"][star_id]*u.second + args.overhead*u.hour)
    if verbose:
        print("t_boundaries", t_boundaries)

    t_boundaries = np.array(t_boundaries)

    inds = np.where(t_boundaries <= Time(args.end, scale="utc"))
    t_boundaries = t_boundaries[inds]
    t_mids = t_boundaries[:-1] + 0.5 * (t_boundaries[1:] - t_boundaries[:-1])

    sequence = np.array(sequence)[:len(t_mids)]


    #print("t_boundaries", t_boundaries, "sequence", sequence, len(sequence))
    #print("t_mids", t_mids)

    #sun_altaz = get_sun(t).transform_to(altaz)

    airmasses = []
    for i in range(len(sequence)):
        airmasses.append(get_airmass(target_list[sequence[i]], t_mids[i]))
    airmasses = np.array(airmasses)
    if verbose:
        print("airmasses", airmasses)

    standards_mask = []
    for star_id in sequence:
        if all_targets["kind"][star_id] == "standard":
            standards_mask.append(1)
        else:
            standards_mask.append(0)
            
    standards_inds = np.where(np.array(standards_mask))
    
    total_score = 0.
    for i in range(len(sequence)):
        if all_targets["kind"][sequence[i]] == "science" and airmasses[i] < args.airmass_max:
            
            unc_squared = get_intercept_unc_squared(airmasses[standards_inds] - airmasses[i])
            unc_squared = 1. + unc_squared # Floor of 1
            total_score += 1./unc_squared
    print("total_score", total_score)
    return total_score





ap = argparse.ArgumentParser(description="Telescope SA scheduler (spectroscopic)")
ap.add_argument('--targets', required=True, help='CSV of candidate visits')
ap.add_argument('--start', required=True, help='UTC start time, e.g. 2025-10-23 18:45:00')
ap.add_argument('--end', required=True, help='UTC end time')
ap.add_argument('--site-lat', type=float, default=19.825)
ap.add_argument('--site-lon', type=float, default=-155.47)
ap.add_argument('--site-elev-m', type=float, default=4205.)
ap.add_argument('--dt-min', type=int, default=2, help='time-grid cadence (minutes)')
ap.add_argument('--airmass-max', type=float, default=2.5)
ap.add_argument('--overhead', type=float, default=0.15, help = 'overhead in hours')

args = ap.parse_args()

all_targets = pd.read_csv(args.targets, skip_blank_lines=True)
print("all_targets", len(all_targets))
print(all_targets)

target_list = []
for i in range(len(all_targets)):
    target_list.append(
        SkyCoord(ra=all_targets["ra_deg"][i]*u.deg, dec=all_targets["dec_deg"][i]*u.deg, frame="icrs")
        )


duration = (Time(args.end, scale="utc") - Time(args.start, scale="utc")).to(u.hour)
print("duration", duration)

max_n_obs = int(duration/((0.05 + args.overhead)*u.hour))
print("max_n_obs", max_n_obs)

maunakea = EarthLocation(lat=args.site_lat, lon=args.site_lon*u.deg, height=args.site_elev_m*u.m)


cfg = SAConfig(
    T_start=2.0,
    T_end=1e-3,
    steps=20,
    iters_per_T=100,
    schedule="geometric",
    maximize=True,
    patience_T=8,
    seed=42)


sa = SimulatedAnnealer(
    fitness=fitness,
    initial_seq = np.random.randint(len(all_targets), size = max_n_obs),
    domain=range(len(all_targets)),
    constraint=None,                     # or your constraint(seq, ...)
    config=cfg,
    # constraint_args=..., constraint_kwargs=... also supported
)

best_seq, best_score, stats = sa.optimize()
print("best_seq, best_score", best_seq, best_score)
fitness(best_seq, verbose=True)
for star_ind in best_seq:
    print("observe", all_targets["id"][star_ind])
